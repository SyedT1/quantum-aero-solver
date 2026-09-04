"""Build the competition concept proposal as a polished PDF."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    KeepTogether,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
TMP = ROOT / "tmp" / "pdfs"
OUT = ROOT / "output" / "pdf" / "quantum_aero_concept_proposal.pdf"

NAVY = colors.HexColor("#071D3B")
BLUE = colors.HexColor("#1261A0")
CYAN = colors.HexColor("#00A6C8")
PALE = colors.HexColor("#EAF5F8")
LIGHT = colors.HexColor("#F3F6F8")
MID = colors.HexColor("#5C6B78")
GREEN = colors.HexColor("#14866D")
AMBER = colors.HexColor("#D98716")
RED = colors.HexColor("#B33A3A")
WHITE = colors.white


def register_fonts() -> tuple[str, str]:
    regular = Path("C:/Windows/Fonts/arial.ttf")
    bold = Path("C:/Windows/Fonts/arialbd.ttf")
    if regular.exists() and bold.exists():
        pdfmetrics.registerFont(TTFont("ProposalSans", regular))
        pdfmetrics.registerFont(TTFont("ProposalSans-Bold", bold))
        return "ProposalSans", "ProposalSans-Bold"
    return "Helvetica", "Helvetica-Bold"


FONT, FONT_BOLD = register_fonts()


def equation(name: str, latex: str, width: float = 6.7, height: float = 0.72) -> Path:
    path = TMP / f"eq_{name}.png"
    fig = plt.figure(figsize=(width, height), dpi=220)
    fig.patch.set_alpha(0)
    fig.text(0.5, 0.5, f"${latex}$", ha="center", va="center", fontsize=17, color="#071D3B")
    fig.savefig(path, transparent=True, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return path


def make_scaling_chart() -> Path:
    path = TMP / "scaling_chart.png"
    rows = list(csv.DictReader((ROOT / "results" / "reynolds_sweep.csv").open(encoding="utf-8")))
    re = np.array([float(r["reynolds"]) for r in rows])
    runtime = np.array([float(r["runtime_median_seconds"]) for r in rows])
    error = np.array([float(r["relative_l2"]) for r in rows])
    pilot = re == 5000

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.1), dpi=190)
    fig.patch.set_facecolor("white")
    axes[0].loglog(re[~pilot], runtime[~pilot], "o-", color="#1261A0", lw=2, ms=6)
    axes[0].loglog(re[pilot], runtime[pilot], "D", color="#D98716", ms=7, label="under-resolved pilot")
    axes[0].set(xlabel="Reynolds number", ylabel="Measured runtime (s)", title="Classical time-to-solution")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].semilogx(re[~pilot], error[~pilot], "o-", color="#14866D", lw=2, ms=6)
    axes[1].semilogx(re[pilot], error[pilot], "D", color="#D98716", ms=7)
    axes[1].set(xlabel="Reynolds number", ylabel="Relative velocity L2", title="Accuracy at t = 1")
    for ax in axes:
        ax.grid(True, which="both", alpha=0.18)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def p(text: str, style, **kwargs) -> Paragraph:
    return Paragraph(text, style, **kwargs)


def bullet(text: str, styles) -> Paragraph:
    return Paragraph(f"<bullet>&bull;</bullet>{text}", styles["BulletBody"])


def section(title: str, styles) -> list:
    return [Spacer(1, 2 * mm), Paragraph(title, styles["H1"]), Spacer(1, 2 * mm)]


def metric_card(value: str, label: str, accent=CYAN) -> Table:
    card = Table(
        [[Paragraph(value, STYLES["MetricValue"])], [Paragraph(label, STYLES["MetricLabel"])]],
        colWidths=[43 * mm], rowHeights=[12 * mm, 12 * mm],
    )
    card.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), LIGHT),
        ("BOX", (0, 0), (-1, -1), 0.7, accent),
        ("LINEABOVE", (0, 0), (-1, 0), 3, accent),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
    ]))
    return card


def architecture_table(styles) -> Table:
    labels = [
        ("1", "Prepare", "Encode populations\nand selected observables"),
        ("2", "Collide", "Carleman lift +\nblock encoding"),
        ("3", "Stream", "Unitary periodic\nD2Q9 permutation"),
        ("4", "Measure", "Energy, modes,\nerror indicators"),
    ]
    cells = []
    for number, title, body in labels:
        cells.append(Paragraph(
            f'<font color="#00A6C8"><b>{number}</b></font><br/><b>{title}</b><br/>'
            f'<font size="8" color="#5C6B78">{body}</font>', styles["Arch"],
        ))
    table = Table([cells], colWidths=[42 * mm] * 4, rowHeights=[31 * mm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), LIGHT),
        ("BOX", (0, 0), (-1, -1), 0.8, CYAN),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#B9DDE5")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]))
    return table


class ProposalDoc(BaseDocTemplate):
    def __init__(self, filename: str):
        super().__init__(
            filename,
            pagesize=A4,
            leftMargin=19 * mm,
            rightMargin=19 * mm,
            topMargin=19 * mm,
            bottomMargin=18 * mm,
            title="Quantum Aero Solver - Concept Proposal",
            author="Mohaimin | Quantum Aero Solver",
            subject="Airbus Global Quantum + AI Challenge 2026",
        )
        frame = Frame(self.leftMargin, self.bottomMargin, self.width, self.height, id="body")
        self.addPageTemplates([
            PageTemplate(id="content", frames=[frame], onPage=self.draw_page),
        ])

    def draw_page(self, canvas, doc):
        page = canvas.getPageNumber()
        if page == 1:
            canvas.saveState()
            canvas.setFillColor(NAVY)
            canvas.rect(0, 0, A4[0], A4[1], fill=1, stroke=0)
            canvas.setFillColor(CYAN)
            canvas.rect(0, A4[1] - 8 * mm, A4[0], 8 * mm, fill=1, stroke=0)
            canvas.restoreState()
            return
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#CAD5DD"))
        canvas.line(19 * mm, 14 * mm, A4[0] - 19 * mm, 14 * mm)
        canvas.setFont(FONT, 7.5)
        canvas.setFillColor(MID)
        canvas.drawString(19 * mm, 9 * mm, "QUANTUM AERO SOLVER | CONCEPT PROPOSAL")
        canvas.drawRightString(A4[0] - 19 * mm, 9 * mm, f"{page}")
        canvas.setFillColor(CYAN)
        canvas.rect(0, A4[1] - 4 * mm, A4[0], 4 * mm, fill=1, stroke=0)
        canvas.restoreState()


def make_styles():
    base = getSampleStyleSheet()
    styles = {
        "CoverKicker": ParagraphStyle(
            "CoverKicker", fontName=FONT_BOLD, fontSize=10, leading=13,
            textColor=colors.HexColor("#7DE0F0"), spaceAfter=8,
        ),
        "CoverTitle": ParagraphStyle(
            "CoverTitle", fontName=FONT_BOLD, fontSize=28, leading=33,
            textColor=WHITE, spaceAfter=8,
        ),
        "CoverSub": ParagraphStyle(
            "CoverSub", fontName=FONT, fontSize=14, leading=20,
            textColor=colors.HexColor("#D8EAF1"), spaceAfter=12,
        ),
        "CoverMeta": ParagraphStyle(
            "CoverMeta", fontName=FONT, fontSize=9, leading=14,
            textColor=colors.HexColor("#AFC9D5"),
        ),
        "H1": ParagraphStyle(
            "H1", fontName=FONT_BOLD, fontSize=18, leading=22, textColor=NAVY,
            spaceBefore=2, spaceAfter=7, keepWithNext=True,
        ),
        "H2": ParagraphStyle(
            "H2", fontName=FONT_BOLD, fontSize=11.5, leading=15, textColor=BLUE,
            spaceBefore=9, spaceAfter=4, keepWithNext=True,
        ),
        "Body": ParagraphStyle(
            "Body", fontName=FONT, fontSize=9.2, leading=13.2, textColor=colors.HexColor("#243342"),
            spaceAfter=6,
        ),
        "Small": ParagraphStyle(
            "Small", fontName=FONT, fontSize=7.6, leading=10.3, textColor=MID,
            spaceAfter=4,
        ),
        "BulletBody": ParagraphStyle(
            "BulletBody", parent=base["BodyText"], fontName=FONT, fontSize=9,
            leading=12.5, textColor=colors.HexColor("#243342"), leftIndent=11,
            firstLineIndent=-7, bulletIndent=2, spaceAfter=4,
        ),
        "Callout": ParagraphStyle(
            "Callout", fontName=FONT, fontSize=9.2, leading=13.5, textColor=NAVY,
            backColor=PALE, borderColor=CYAN, borderWidth=0.8, borderPadding=8,
            spaceBefore=5, spaceAfter=8,
        ),
        "MetricValue": ParagraphStyle(
            "MetricValue", fontName=FONT_BOLD, fontSize=16, leading=18,
            textColor=NAVY, alignment=TA_CENTER,
        ),
        "MetricLabel": ParagraphStyle(
            "MetricLabel", fontName=FONT, fontSize=7.5, leading=9.5,
            textColor=MID, alignment=TA_CENTER,
        ),
        "Arch": ParagraphStyle(
            "Arch", fontName=FONT, fontSize=9.5, leading=13,
            textColor=NAVY, alignment=TA_CENTER,
        ),
        "TableHead": ParagraphStyle(
            "TableHead", fontName=FONT_BOLD, fontSize=7.6, leading=9.5,
            textColor=WHITE,
        ),
        "TableCell": ParagraphStyle(
            "TableCell", fontName=FONT, fontSize=7.4, leading=9.5,
            textColor=colors.HexColor("#243342"),
        ),
        "Ref": ParagraphStyle(
            "Ref", fontName=FONT, fontSize=7.3, leading=10.2, textColor=colors.HexColor("#34495A"),
            leftIndent=10, firstLineIndent=-10, spaceAfter=4,
        ),
    }
    return styles


STYLES = make_styles()


def styled_table(data, widths, header=True, row_bgs=None) -> Table:
    cooked = []
    for r, row in enumerate(data):
        style = STYLES["TableHead"] if header and r == 0 else STYLES["TableCell"]
        cooked.append([Paragraph(str(cell), style) for cell in row])
    table = Table(cooked, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    commands = [
        ("BACKGROUND", (0, 0), (-1, 0), NAVY if header else LIGHT),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#C7D4DC")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]
    if row_bgs:
        for row, color in row_bgs.items():
            commands.append(("BACKGROUND", (0, row), (-1, row), color))
    table.setStyle(TableStyle(commands))
    return table


def build_story() -> list:
    validation = json.loads((ROOT / "results" / "validation.json").read_text(encoding="utf-8"))
    story = [NextPageTemplate("content")]

    # Cover
    story += [
        Spacer(1, 40 * mm),
        Paragraph("AIRBUS GLOBAL QUANTUM + AI CHALLENGE 2026", STYLES["CoverKicker"]),
        Paragraph("Quantum Aero Solver", STYLES["CoverTitle"]),
        Paragraph(
            "A block-encoded Carleman-LBM concept for the convecting "
            "Taylor-Green vortex", STYLES["CoverSub"],
        ),
        Spacer(1, 8 * mm),
        Table([["CONCEPT PROPOSAL", "TRACK A | FTQC / LBM"]], colWidths=[55 * mm, 67 * mm], style=[
            ("FONTNAME", (0, 0), (-1, -1), FONT_BOLD),
            ("FONTSIZE", (0, 0), (-1, -1), 8.5),
            ("TEXTCOLOR", (0, 0), (-1, -1), WHITE),
            ("BACKGROUND", (0, 0), (0, 0), BLUE),
            ("BACKGROUND", (1, 0), (1, 0), CYAN),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ]),
        Spacer(1, 55 * mm),
        Paragraph(
            "Prepared by: Mohaimin | Team Quantum Aero Solver<br/>"
            "Date: 4 September 2026<br/>"
            "Status: validated concept; quantum advantage not yet claimed",
            STYLES["CoverMeta"],
        ),
        PageBreak(),
    ]

    # Executive summary
    story += section("1. Executive summary", STYLES)
    story += [
        p(
            "Airbus seeks a more scalable route to high-fidelity aerodynamic simulation at "
            "conditions where classical computation and physical testing become costly. We propose "
            "a fault-tolerant quantum Lattice Boltzmann solver for the two-dimensional convecting "
            "Taylor-Green vortex. The concept combines a D2Q9 finite-velocity representation, "
            "Carleman linearization of the nonlinear BGK collision, a block encoding of the resulting "
            "non-unitary map, and exact unitary streaming on a periodic grid.", STYLES["Body"],
        ),
        p(
            "The proposal is deliberately end-to-end: advantage will be assessed at fixed physical "
            "accuracy and will include state preparation, time stepping, block-encoding normalization, "
            "amplitude amplification, observable extraction, routing, and fault-tolerant overhead. "
            "The target output is not the full velocity field; it is a small set of engineering "
            "observables such as kinetic-energy decay and selected Fourier modes, avoiding the data-"
            "extraction bottleneck that can erase a quantum speedup.", STYLES["Body"],
        ),
        Paragraph(
            "<b>Core hypothesis.</b> For high Reynolds number and selected low-dimensional "
            "observables, a structured implementation of collision plus streaming can reach a lower "
            "asymptotic or crossover time-to-solution than a grid-converged classical reference at the "
            "same error tolerance. The project includes an explicit no-go gate if end-to-end resource "
            "accounting does not support that hypothesis.", STYLES["Callout"],
        ),
        Spacer(1, 2 * mm),
        Table([[metric_card("90", "lifted collision dimension"),
                metric_card("8", "padded local logical qubits", BLUE),
                metric_card("1.77%", "raw post-selection", AMBER)]],
              colWidths=[56 * mm] * 3, style=[("VALIGN", (0, 0), (-1, -1), "TOP")]),
        Spacer(1, 5 * mm),
        p("Proposed contribution", STYLES["H2"]),
        bullet("A corrected, periodic, analytically verifiable TGV benchmark with complete provenance.", STYLES),
        bullet("A physically meaningful nine-population collision map rather than a toy direction-bit gate.", STYLES),
        bullet("A structured block-encoding research path benchmarked against a dense upper bound.", STYLES),
        bullet("A transparent advantage claim based on selected observables and total cost, not qubit count alone.", STYLES),
        PageBreak(),
    ]

    # Problem
    story += section("2. Benchmark and physical model", STYLES)
    story += [
        p(
            "The benchmark is the incompressible convecting Taylor-Green vortex on a periodic "
            "two-dimensional domain. It is challenging enough to expose nonlinear advection, viscous "
            "dissipation, numerical dispersion, and stability limits, while retaining an exact solution "
            "for rigorous fidelity measurement [1].", STYLES["Body"],
        ),
        Image(str(equation("ns", r"\nabla\!\cdot\!\mathbf{u}=0,\qquad \partial_t\mathbf{u}+(\mathbf{u}\!\cdot\!\nabla)\mathbf{u}=-\rho^{-1}\nabla p+\nu\nabla^2\mathbf{u}")), width=168 * mm, height=18 * mm),
        p("We separate the periodic box length from the vortex length scale:", STYLES["Body"]),
        Image(str(equation("exact_u", r"u=U_c+V_0\sin\!\left(\frac{x-U_ct}{L_c}\right)\cos\!\left(\frac{y-V_ct}{L_c}\right)e^{-2\nu t/L_c^2}")), width=160 * mm, height=16 * mm),
        Image(str(equation("exact_v", r"v=V_c-V_0\cos\!\left(\frac{x-U_ct}{L_c}\right)\sin\!\left(\frac{y-V_ct}{L_c}\right)e^{-2\nu t/L_c^2}")), width=160 * mm, height=16 * mm),
        Image(str(equation("exact_p", r"p=p_0+\frac{\rho V_0^2}{4}\left[\cos\!\left(\frac{2(x-U_ct)}{L_c}\right)+\cos\!\left(\frac{2(y-V_ct)}{L_c}\right)\right]e^{-4\nu t/L_c^2}")), width=168 * mm, height=17 * mm),
        Spacer(1, 2 * mm),
        styled_table([
            ["Parameter", "Value", "Interpretation"],
            ["Periodic domain", "[0, 2 pi) x [0, 2 pi)", "L_box = 2 pi"],
            ["Vortex scale", "L_c = 1", "Makes the exact field periodic"],
            ["Velocity", "V0 = 1, Uc = 1, Vc = 0", "Unit vortex in streamwise flow"],
            ["Density / pressure", "rho = 1, p0 = 0", "Constant-density benchmark"],
            ["Viscosity", "nu = V0 L_c / Re", "Re = 10 through 5000+"],
            ["Primary final time", "t = 1", "Matches the challenge illustration"],
        ], [38 * mm, 54 * mm, 76 * mm]),
        Spacer(1, 4 * mm),
        Paragraph(
            "<b>Benchmark clarification.</b> The challenge statement uses the same symbol L for a "
            "2 pi domain and for the trigonometric vortex scale. Taken literally, that field is not "
            "periodic. We use L_box = 2 pi and L_c = 1, state the convention in every artifact, and "
            "will request organizer confirmation before final scoring.", STYLES["Callout"],
        ),
        PageBreak(),
    ]

    # Classical + LBM
    story += section("3. Numerical foundation: D2Q9 LBM", STYLES)
    story += [
        p(
            "The classical control and quantum formulation share the same D2Q9 discretization. "
            "This removes discretization mismatch from the first quantum-versus-classical comparison. "
            "At each cell, nine populations encode density and momentum:", STYLES["Body"],
        ),
        Image(str(equation("macro", r"\rho=\sum_{i=0}^{8}f_i,\qquad \rho\mathbf{u}=\sum_{i=0}^{8}f_i\mathbf{e}_i,\qquad c_s^2=\frac{1}{3}")), width=140 * mm, height=16 * mm),
        Image(str(equation("feq", r"f_i^{eq}=w_i\rho\left[1+\frac{\mathbf{e}_i\cdot\mathbf{u}}{c_s^2}+\frac{(\mathbf{e}_i\cdot\mathbf{u})^2}{2c_s^4}-\frac{\mathbf{u}\cdot\mathbf{u}}{2c_s^2}\right]")), width=160 * mm, height=18 * mm),
        Image(str(equation("bgk", r"f_i^{\star}=f_i-\omega(f_i-f_i^{eq}),\qquad f_i(\mathbf{x}+\mathbf{e}_i,t+\Delta t)=f_i^{\star}(\mathbf{x},t),\qquad \omega=\tau^{-1}")), width=166 * mm, height=18 * mm),
        p("Validation metrics", STYLES["H2"]),
        Image(str(equation("l2", r"\epsilon_{L2}=\frac{\left\|\mathbf{u}_{sim}-\mathbf{u}_{exact}\right\|_2}{\left\|\mathbf{u}_{exact}\right\|_2},\qquad K(t)=\frac{1}{2}\left\langle u^2+v^2\right\rangle")), width=145 * mm, height=16 * mm),
        bullet("Total-field and vortex-only velocity L2 error, to prevent the uniform stream from hiding vortex error.", STYLES),
        bullet("Kinetic-energy decay, mass drift, momentum drift, and divergence norm.", STYLES),
        bullet("Grid, Mach, and timestep convergence before any advantage comparison.", STYLES),
        p("Classical comparators", STYLES["H2"]),
        p(
            "The in-repository BGK solver is the controlled algorithmic baseline, not the "
            "state-of-the-art comparator. The final study will add a Fourier pseudo-spectral or "
            "high-order finite-volume reference, use identical hardware-reporting rules, and compare "
            "time-to-solution at fixed error rather than at fixed grid size.", STYLES["Body"],
        ),
        PageBreak(),
    ]

    # Quantum algorithm
    story += section("4. Proposed fault-tolerant quantum algorithm", STYLES)
    story += [
        architecture_table(STYLES),
        Spacer(1, 5 * mm),
        p("4.1 Carleman collision", STYLES["H2"]),
        p(
            "Around reference density rho0, the quadratic equilibrium is written as a linear and "
            "quadratic function of the nine populations. Define the order-2 lifted state and collision "
            "matrix:", STYLES["Body"],
        ),
        Image(str(equation("carleman", r"f^{eq}\approx Lf+Q:(f\otimes f),\qquad z=(f,\ f\otimes f)^T,\qquad z^{\star}=Mz")), width=145 * mm, height=18 * mm),
        Image(str(equation("matrix_m", r"f^{\star}=Rf+\omega Q:(f\otimes f),\qquad (f\otimes f)^{\star}\approx(R\otimes R)(f\otimes f),\qquad R=(1-\omega)I+\omega L")), width=165 * mm, height=18 * mm),
        p(
            "For one cell, z has dimension 9 + 81 = 90. This is padded to 128 amplitudes. "
            "The non-unitary collision is normalized by alpha >= ||M||2 and embedded into a unitary:",
            STYLES["Body"],
        ),
        Image(str(equation("dilation", r"A=M/\alpha,\quad U_A^{00}=A,\quad U_A^{01}=\sqrt{I-AA^{\dagger}},\quad U_A^{10}=\sqrt{I-A^{\dagger}A},\quad U_A^{11}=-A^{\dagger}")), width=168 * mm, height=18 * mm),
        p(
            "Post-selecting the signal ancilla applies A. Oblivious amplitude amplification is included "
            "in the cost model; success probability is never treated as free.", STYLES["Body"],
        ),
        p("4.2 Streaming and observables", STYLES["H2"]),
        p(
            "Streaming is a reversible, direction-controlled modular shift over the position register. "
            "The primary outputs are mean kinetic energy and selected Fourier coefficients. Full-field "
            "tomography is excluded from the claimed advantage path because its extraction cost scales "
            "with the field dimension.", STYLES["Body"],
        ),
        Paragraph(
            "<b>Research gate:</b> the present spatial validator rebuilds f tensor f classically. "
            "Milestone 2 must replace that step with a coherent global/local lift or adopt an end-to-end "
            "incompressible LBM construction with bounded advantage [4]. Failure triggers a no-go result, "
            "not an unsupported speedup claim.", STYLES["Callout"],
        ),
        PageBreak(),
    ]

    # Evidence
    block = validation["block_encoding"]
    collision = validation["collision_resources"]
    noise = validation["noise"]
    story += section("5. Preliminary evidence and technical readiness", STYLES)
    story += [
        p(
            "The corrected repository provides executable evidence for each local building block and "
            "records limitations alongside results. These figures are feasibility evidence, not a "
            "quantum-advantage claim.", STYLES["Body"],
        ),
        Image(str(make_scaling_chart()), width=170 * mm, height=60 * mm),
        Spacer(1, 3 * mm),
        styled_table([
            ["Experiment", "Measured result", "Interpretation"],
            ["Canonical datasets", "Final L2 = 0.00443 to 0.00463", "Legacy 25-33% error removed"],
            ["Collision block", f"max error = {block['max_block_error']:.2e}", "90D map reproduced numerically"],
            ["Dilation unitarity", f"max error = {block['max_unitarity_error']:.2e}", "8-qubit dilation verified"],
            ["Post-selection", f"p = {block['postselection_probability']:.4f}", f"{block['expected_postselection_attempts']:.1f} raw attempts expected"],
            ["Spatial fidelity", "velocity L2 = 1.28e-5", "Carleman versus exact BGK, N=32, t=0.1"],
            ["Applied noise", f"Hellinger fidelity = {noise['hellinger_fidelity']:.3f}", "20,000 shots; noise actually passed to Aer"],
            ["Routed dense synthesis", f"depth {collision['transpiled_depth']:,}; CX {collision['transpiled_operations']['cx']:,}", "Upper-bound baseline; structure is essential"],
            ["Re=5000 pilot", "N=128, t=1, 65.5 s", "Measured but explicitly under-resolved"],
        ], [39 * mm, 53 * mm, 76 * mm], row_bgs={4: colors.HexColor("#FFF5E5"), 7: colors.HexColor("#FFF5E5")}),
        Spacer(1, 5 * mm),
        p(
            "The dense routed collision requires 51,483 one-qubit u gates and 87,216 CX gates on a "
            "nearest-neighbor ring. This is not the proposed production circuit; it is a measured upper "
            "bound that establishes the optimization target. A structured implementation must exploit "
            "D2Q9 sparsity, repeated coefficients, symmetry, and local collision reuse.", STYLES["Body"],
        ),
        PageBreak(),
    ]

    # Experiments
    story += section("6. Experimental program", STYLES)
    story += [
        p(
            "Every comparison uses the same physical domain, final time, Reynolds number, observable, "
            "and error tolerance. Each timing point receives warm-up runs and at least five measured "
            "repetitions for the final submission.", STYLES["Body"],
        ),
        styled_table([
            ["Work package", "Sweep", "Recorded outputs", "Pass criterion"],
            ["A. Physics convergence", "Re 10, 100, 400, 1000, 2000, 5000; N 32-2048; Ma 0.1-0.0125", "L2, energy, phase, mass, divergence", "Grid and Mach convergence demonstrated"],
            ["B. Collision fidelity", "Carleman order 1-3; 1-1000 steps; density/velocity perturbations", "Population and observable drift", "Error budget below target epsilon"],
            ["C. Quantum correctness", "1, 2, 5, 10, 50 timesteps", "State/observable error versus BGK", "No hidden classical operation in claimed path"],
            ["D. Noise sensitivity", "Gate error 1e-4 to 1e-2; shots 1e3 to 1e6", "Fidelity, bias, confidence interval", "Noise model applied after transpilation"],
            ["E. Resource scaling", "N 8-2048; Re 10-5000; epsilon 1e-1 to 1e-4", "Logical/physical qubits, T count, depth, calls", "Complete cost ledger"],
            ["F. Classical reference", "BGK plus spectral/high-order FV", "Runtime, peak memory, accuracy", "Same tolerance and reporting protocol"],
        ], [34 * mm, 49 * mm, 47 * mm, 38 * mm]),
        Spacer(1, 5 * mm),
        p("Required figures for the final submission", STYLES["H2"]),
        bullet("Time-to-solution versus Reynolds number at fixed error, with confidence intervals.", STYLES),
        bullet("Memory or physical-qubit requirement versus Reynolds number.", STYLES),
        bullet("Velocity L2 and kinetic-energy error versus Reynolds number and resolution.", STYLES),
        bullet("Total quantum cost broken into preparation, collision, streaming, amplification, and measurement.", STYLES),
        bullet("Crossover or no-crossover plot against both classical comparators.", STYLES),
        p("Reproducibility", STYLES["H2"]),
        p(
            "All configurations, seeds, package versions, CPU/GPU/backend specifications, transpiler "
            "settings, raw counts, and derived metrics are machine-readable. Failed and null experiments "
            "remain in the evidence trail.", STYLES["Body"],
        ),
        PageBreak(),
    ]

    # Advantage
    story += section("7. Definition and test of quantum advantage", STYLES)
    story += [
        p(
            "A smaller amplitude register is not, by itself, an advantage. The claim is accepted only "
            "if total fault-tolerant time or memory is lower at the same physical error for a declared "
            "observable. We therefore use the following cost ledger:", STYLES["Body"],
        ),
        Image(str(equation("tts", r"T_Q=T_{prep}+n_t\left[N_{BE}C_{AA}(p)+N_S\right]t_{cycle}+N_{shots}T_{meas}")), width=145 * mm, height=17 * mm),
        p(
            "Here N_BE is the block-encoding cost, C_AA(p) captures post-selection or amplitude "
            "amplification, N_S is streaming cost, and N_shots is set by the observable precision. "
            "Rotation synthesis, routing, magic-state factories, logical cycle time, and failure "
            "probability are included in t_cycle for each hardware scenario.", STYLES["Body"],
        ),
        styled_table([
            ["Decision gate", "Evidence required", "Action"],
            ["G1 - physical fidelity", "Converged classical TGV and stable Carleman error", "Proceed only if error budget closes"],
            ["G2 - coherent timestep", "No classical re-lifting inside claimed evolution", "Proceed to FT resource compilation"],
            ["G3 - structured collision", "Material reduction from dense 107,128 depth", "Reject dense implementation if no structure found"],
            ["G4 - end-to-end crossover", "T_Q < T_C or M_Q < M_C at fixed epsilon", "Claim bounded advantage, otherwise report no-go"],
        ], [35 * mm, 77 * mm, 56 * mm], row_bgs={4: colors.HexColor("#EAF7F1")}),
        Spacer(1, 6 * mm),
        Paragraph(
            "<b>Primary claim boundary.</b> The proposal targets bounded advantage for selected "
            "observables at useful error tolerance. It does not promise exponential advantage for "
            "full-field recovery. Recent end-to-end analysis identifies convergence, condition number, "
            "time stepping, and data extraction as decisive bottlenecks [4]; our gates test each one.",
            STYLES["Callout"],
        ),
        p("Hardware scenarios", STYLES["H2"]),
        bullet("Logical all-to-all: algorithmic scaling and lower-bound comparison.", STYLES),
        bullet("Routed logical: nearest-neighbor connectivity and native basis decomposition.", STYLES),
        bullet("Fault tolerant: surface-code distance, factory throughput, physical qubits, and wall time.", STYLES),
        PageBreak(),
    ]

    # Workplan risks
    story += section("8. Delivery plan, risks, and impact", STYLES)
    story += [
        styled_table([
            ["Phase", "Duration", "Deliverable", "Exit evidence"],
            ["1. Benchmark lock", "Weeks 1-2", "Canonical equations, data, spectral/FV comparator", "Convergence and provenance report"],
            ["2. Coherent algorithm", "Weeks 3-6", "Global/local lifted timestep without classical re-lift", "Multi-step state/observable fidelity"],
            ["3. Circuit engineering", "Weeks 7-10", "Sparse structured block encoding and streaming", "Native-basis and routed resources"],
            ["4. Scale study", "Weeks 11-13", "Re and epsilon sweeps", "Runtime, memory, error plots"],
            ["5. FT assessment", "Weeks 14-15", "Physical resource scenarios", "Crossover or no-go conclusion"],
            ["6. Submission", "Week 16", "Technical report, code, datasets, reproducibility bundle", "Independent rerun checklist"],
        ], [34 * mm, 24 * mm, 60 * mm, 50 * mm]),
        Spacer(1, 6 * mm),
        p("Top risks and mitigations", STYLES["H2"]),
        styled_table([
            ["Risk", "Likelihood / impact", "Mitigation"],
            ["Carleman truncation fails at high Re", "High / high", "Adaptive order, rescaling, stability map; pivot to bounded-advantage incompressible LBM"],
            ["Post-selection cost dominates", "High / high", "Normalization optimization, amplitude amplification, local encodings; explicit stop gate"],
            ["Dense synthesis remains exponential", "High / high", "Exploit D2Q9 sparsity and symmetry; quantify structure before scale claims"],
            ["Data extraction removes speedup", "High / high", "Restrict outputs to energy and selected modes; compare full observable cost"],
            ["Classical reference is too weak", "Medium / high", "Add spectral/high-order FV comparator and fixed-error methodology"],
            ["PDF parameter ambiguity", "Medium / medium", "Separate L_box and L_c; request organizer confirmation"],
        ], [48 * mm, 35 * mm, 85 * mm]),
        Spacer(1, 6 * mm),
        p("Expected impact", STYLES["H2"]),
        p(
            "The project will deliver either a defensible bounded quantum advantage for selected "
            "aerodynamic observables or a quantified no-go boundary showing exactly which overhead "
            "prevents it. Both outcomes reduce uncertainty for future Airbus quantum-CFD investment. "
            "The validated benchmark, resource ledger, and reproducible datasets remain useful beyond "
            "the challenge for solver selection and hardware-roadmap studies.", STYLES["Body"],
        ),
        PageBreak(),
    ]

    # References
    story += section("9. References and evidence artifacts", STYLES)
    refs = [
        "[1] Airbus. Quantum Solvers: Enhancing Predictive Aerodynamic Modeling Capabilities. Global Quantum + AI Challenge 2026 Enterprise Challenge Statement, 2026.",
        "[2] Y. H. Qian, D. d'Humieres, and P. Lallemand. Lattice BGK Models for Navier-Stokes Equation. Europhysics Letters 17(6), 1992.",
        "[3] J.-P. Liu et al. Efficient quantum algorithm for dissipative nonlinear differential equations. PNAS 118, e2026805118, 2021; corrected arXiv version 2026. https://arxiv.org/abs/2011.03185",
        "[4] D. Jennings et al. An end-to-end quantum algorithm for nonlinear fluid dynamics with bounded quantum advantage. arXiv:2512.03758, 2025. https://arxiv.org/abs/2512.03758",
        "[5] A. D. Bastida Zamora et al. Quantum lattice Boltzmann method for several time steps: A local Carleman linearization algorithm. arXiv:2511.13072v3, 2026. https://arxiv.org/abs/2511.13072",
        "[6] A. Gilyen, Y. Su, G. H. Low, and N. Wiebe. Quantum singular value transformation and beyond. STOC 2019. https://arxiv.org/abs/1806.01838",
        "[7] A. M. Childs, R. Kothari, and R. D. Somma. Quantum algorithm for systems of linear equations with exponentially improved dependence on precision. SIAM Journal on Computing 46(6), 2017.",
    ]
    story += [Paragraph(ref, STYLES["Ref"]) for ref in refs]
    story += [
        Spacer(1, 7 * mm),
        p("Repository evidence", STYLES["H2"]),
        styled_table([
            ["Artifact", "Purpose"],
            ["quantum_aero/classical.py", "Corrected periodic D2Q9 solver and diagnostics"],
            ["quantum_aero/carleman.py", "Nine-population Carleman map and unitary dilation"],
            ["quantum_aero/quantum.py", "Applied-noise experiment and routed resource counts"],
            ["results/validation.json", "Machine-readable block, noise, fidelity, and resource evidence"],
            ["results/reynolds_sweep.csv", "Measured t=1 Reynolds sweep including under-resolved Re=5000 pilot"],
            ["baseline/dataset/*.h5", "Corrected 21-snapshot datasets with embedded provenance"],
        ], [64 * mm, 104 * mm]),
        Spacer(1, 8 * mm),
        Paragraph(
            "<b>Submission statement.</b> Preliminary results establish correctness of the local "
            "collision block and benchmark pipeline. They do not yet establish an end-to-end quantum "
            "advantage. The experimental gates in this proposal define the evidence required before "
            "such a claim will be made.", STYLES["Callout"],
        ),
    ]
    return story


def main() -> None:
    TMP.mkdir(parents=True, exist_ok=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = ProposalDoc(str(OUT))
    doc.build(build_story())
    print(OUT)


if __name__ == "__main__":
    main()
