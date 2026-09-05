"""Prepare or explicitly submit the small Open Quantum streaming validation.

Credentials are read from the standard SDK environment variables or prompted
with ``getpass``. They are never written to disk. Preparing a job obtains a
quote; actual QPU submission requires the explicit ``--submit`` flag.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from getpass import getpass
import json
import os
from pathlib import Path

from openquantum_sdk.auth import ClientCredentials, ClientCredentialsAuth
from openquantum_sdk.clients import ManagementClient, SchedulerClient
from openquantum_sdk.models import JobCreate, JobPreparationCreate
from openquantum_sdk.utils import poll_for_status


ROOT = Path(__file__).resolve().parents[1]


def authentication() -> ClientCredentialsAuth:
    client_id = os.getenv("OPENQUANTUM_CLIENT_ID") or input("Client ID: ")
    client_secret = os.getenv("OPENQUANTUM_CLIENT_SECRET") or getpass("Client secret: ")
    return ClientCredentialsAuth(
        creds=ClientCredentials(client_id=client_id, client_secret=client_secret)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="iqm:emerald")
    parser.add_argument("--shots", type=int, default=100)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()

    auth = authentication()
    management = ManagementClient(auth=auth)
    scheduler = SchedulerClient(auth=auth)
    try:
        organization = management.list_user_organizations().organizations[0]
        balance = management.get_credit_balance(organization.id)
        qasm = (ROOT / "hardware" / "openquantum_streaming.qasm").read_bytes()
        upload_id = scheduler.upload_job_input(file_content=qasm)
        preparation = scheduler.prepare_job(
            JobPreparationCreate(
                organization_id=organization.id,
                backend_class_id=args.backend,
                name="Quantum Aero controlled streaming validation",
                upload_endpoint_id=upload_id,
                job_subcategory_id="math:deq",
                shots=args.shots,
                configuration_data={"shots": args.shots},
                submitted_with="sdk",
                input_format="qasm",
            )
        )

        def preparation_status(preparation_id: str):
            result = scheduler.get_preparation_result(preparation_id)
            return result.status in ("Completed", "Failed"), result

        result = poll_for_status(
            get_status_fn=preparation_status,
            resource_id=preparation.id,
            interval=2.0,
            timeout=300,
        )
        quotes = [asdict(plan) for plan in result.quote]
        safe = {
            "backend": args.backend,
            "shots": args.shots,
            "preparation_status": result.status,
            "preparation_message": result.message,
            "balance": {
                "spark_credits": balance.spark_credits,
                "full_credits": balance.full_credits,
            },
            "quotes": quotes,
            "submitted": False,
        }
        print(json.dumps(safe, indent=2))
        output = ROOT / "results" / "hardware"
        output.mkdir(parents=True, exist_ok=True)
        (output / "openquantum_streaming_quote.json").write_text(
            json.dumps(safe, indent=2), encoding="utf-8"
        )

        if result.status == "Failed":
            raise RuntimeError(result.message or "Open Quantum preparation failed")
        if not args.submit:
            print("Prepared only. No QPU job was submitted.")
            return
        if not result.quote:
            raise RuntimeError("No execution plan was quoted")

        plan = min(result.quote, key=lambda item: item.price)
        priority = min(plan.queue_priorities, key=lambda item: item.price_increase)
        total_price = plan.price + priority.price_increase
        available = balance.spark_credits + balance.full_credits
        if total_price > available:
            raise RuntimeError(
                f"Insufficient credits: cheapest quote costs {total_price}, balance is {available}"
            )
        job = scheduler.create_job(
            JobCreate(
                job_preparation_id=preparation.id,
                execution_plan_id=plan.execution_plan_id,
                queue_priority_id=priority.queue_priority_id,
            )
        )

        def job_status(job_id: str):
            current = scheduler.get_job(job_id)
            return current.status in ("Completed", "Failed", "Canceled"), current

        final = poll_for_status(
            get_status_fn=job_status,
            resource_id=job.id,
            interval=5.0,
            timeout=86_400,
        )
        safe["submitted"] = True
        safe["job_id"] = final.id
        safe["job_status"] = final.status
        if final.output_data_url:
            safe["output"] = scheduler.download_job_output(final)
        if final.calibration_data_url:
            safe["calibration"] = scheduler.download_job_calibration(final)
        (output / "openquantum_streaming_result.json").write_text(
            json.dumps(safe, indent=2), encoding="utf-8"
        )
        print(json.dumps(safe, indent=2))
    finally:
        scheduler.close()
        management.close()


if __name__ == "__main__":
    main()
