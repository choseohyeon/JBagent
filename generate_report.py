"""
LifeLong WM AI Agent — Quantitative Analysis Report  (English)
JB Financial Group Fin:AI Challenge 2026

6 pages:
  1. Project Overview & Architecture
  2. Survival Analysis — Methodology
  3. Monte Carlo — Methodology
  4. Statistical Robustness & Model Validation
  5. Simulation Results
  6. Assumptions, Limitations & Rebuttals
"""
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
from scipy import stats
from scipy.stats import norm
import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

matplotlib.rcParams.update({
    'font.family':        'DejaVu Sans',
    'axes.unicode_minus': False,
    'font.size':          9,
    'axes.titlesize':     10,
    'axes.labelsize':     9,
    'xtick.labelsize':    8,
    'ytick.labelsize':    8,
    'legend.fontsize':    8,
})

from models.survival    import SurvivalModel, PersonalProfile
from models.monte_carlo import DataLoader, MonteCarloSimulator, SimulationInput

NAVY  = '#1B3A6B'
TEAL  = '#2196A6'
CORAL = '#E05C3A'
GOLD  = '#F0A500'
GREEN = '#27AE60'
LGRAY = '#F4F6F8'
DGRAY = '#4A4A4A'
PURP  = '#7B2D8B'

np.random.seed(42)

# ── Pre-compute (shared across pages) ─────────────────────────────────────────
print("Loading data and running simulations...")
loader   = DataLoader()
params   = loader.fit_distributions()
sim      = MonteCarloSimulator(params)
survival = SurvivalModel()

stock_m  = loader._load_stock_returns()
bond_m   = loader._load_ktb_returns()
cpi_raw  = loader._load_cpi()

stock_a  = (1 + stock_m).resample('YE').prod() - 1
stock_a  = stock_a[stock_a.index.year < 2026]
bond_a   = (1 + bond_m).resample('YE').prod() - 1
bond_a   = bond_a[bond_a.index.year < 2026]
infl_ts  = cpi_raw.pct_change(12).dropna() * 100

stock_kurt = float(stats.kurtosis(stock_a, fisher=True))
stock_skew = float(stats.skew(stock_a))
_, sw_p    = stats.shapiro(stock_a)

CASE_A = SimulationInput(PersonalProfile('M', 60), 30_000, 250, 100, 65, 0.4, 4_000)
CASE_B = SimulationInput(PersonalProfile('F', 60), 50_000, 300, 150, 65, 0.3, 4_000)
RES_A  = sim.run(CASE_A)
RES_B  = sim.run(CASE_B)

# Sequence-of-returns risk (Case A parameters, 2,000 paths)
SEQ_RISK = sim.sequence_risk_report(
    SimulationInput(PersonalProfile('M', 60), 30_000, 250, 100, 65, 0.4),
    front_years=5, n_sims=2_000,
)

# Old fixed-lambda for comparison (lambda=0.08 constant)
class _OldSim(MonteCarloSimulator):
    def _sample_medical_shocks(self, n_years, n_sims, base_age=65):
        counts = np.random.poisson(0.08, (n_sims, n_years))
        sigma  = np.sqrt(np.log(1 + (self.MEDICAL_STD / self.MEDICAL_MEAN)**2))
        mu     = np.log(self.MEDICAL_MEAN) - sigma**2 / 2
        return counts * np.random.lognormal(mu, sigma, (n_sims, n_years))

np.random.seed(42)
old_sim = _OldSim(params)
OLD_A   = old_sim.run(SimulationInput(PersonalProfile('M', 60), 30_000, 250, 100, 65, 0.4, 4_000))

# Sensitivity analysis (n_sims=1500 for speed)
def _dp(assets, expense, pension, stock, seed=42):
    np.random.seed(seed)
    r = sim.run(SimulationInput(PersonalProfile('M', 60), assets, expense, pension, 65, stock, 1500))
    return r.depletion_prob

BASE_DP = _dp(30_000, 250, 100, 0.4)
SENS = {
    'Assets +30%':      _dp(39_000, 250, 100, 0.4),
    'Assets -30%':      _dp(21_000, 250, 100, 0.4),
    'Expense -20%':     _dp(30_000, 200, 100, 0.4),
    'Expense +20%':     _dp(30_000, 300, 100, 0.4),
    'Pension +30%':     _dp(30_000, 250, 130, 0.4),
    'Pension -30%':     _dp(30_000, 250,  70, 0.4),
    'Stock ratio +0.2': _dp(30_000, 250, 100, 0.6),
    'Stock ratio -0.2': _dp(30_000, 250, 100, 0.2),
}
print("Simulations complete.\n")


# ── Helpers ───────────────────────────────────────────────────────────────────
def add_header(fig, title, subtitle=''):
    fig.patch.set_facecolor('white')
    ax = fig.add_axes([0, 0.935, 1, 0.065])
    ax.set_facecolor(NAVY)
    ax.axis('off')
    ax.text(0.02, 0.52, title, color='white', fontsize=12.5,
            fontweight='bold', va='center', transform=ax.transAxes)
    if subtitle:
        ax.text(0.98, 0.52, subtitle, color='#AABBDD', fontsize=8.5,
                va='center', ha='right', transform=ax.transAxes)


def section_label(ax, x, y, text, fontsize=10):
    ax.text(x, y, text, transform=ax.transAxes,
            fontsize=fontsize, fontweight='bold', color=NAVY, va='top')
    ax.axhline(y=y - 0.003, xmin=x, xmax=1.0, color=NAVY, linewidth=1.0,
               transform=ax.transAxes)


def cbox(ax, x, y, w, h, title, lines, bg='#EEF2F8', tc=NAVY, fs=8.2):
    ax.add_patch(mpatches.FancyBboxPatch(
        (x, y), w, h, boxstyle='round,pad=0.01',
        facecolor=bg, edgecolor='#BBBBBB', linewidth=0.8,
        transform=ax.transAxes, clip_on=False))
    ax.text(x+0.012, y+h-0.013, title, transform=ax.transAxes,
            fontsize=9, fontweight='bold', color=tc, va='top')
    for i, ln in enumerate(lines):
        ax.text(x+0.018, y+h-0.060-i*0.040, ln,
                transform=ax.transAxes, fontsize=fs, color=DGRAY, va='top')


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1: PROJECT OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
def page_overview(pdf):
    fig = plt.figure(figsize=(11, 8.5))
    add_header(fig,
               'LifeLong WM AI Agent — Quantitative Analysis Report',
               'JB Financial Group  Fin:AI Challenge 2026')

    ax = fig.add_axes([0.02, 0.02, 0.96, 0.90])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')

    # ── Title block
    ax.text(0.5, 0.97, 'Project Overview', fontsize=14, fontweight='bold',
            color=NAVY, ha='center', va='top')
    ax.axhline(0.94, color=NAVY, linewidth=1.5, xmin=0.01, xmax=0.99)

    # ── Core concept
    ax.text(0.01, 0.91, 'Core Differentiator', fontsize=10, fontweight='bold', color=NAVY)
    ax.text(0.01, 0.87,
            'Conventional WM services express future wealth as a single deterministic number.\n'
            'LifeLong WM expresses it as a probability distribution — quantifying uncertainty\n'
            'that conventional tools deliberately hide from clients.',
            fontsize=8.8, color=DGRAY, va='top')

    for x, label, example in [
        (0.05, 'CONVENTIONAL',  '"You will run out of money at age 83."'),
        (0.52, 'THIS SERVICE',  '"Probability of asset depletion before 83:  41%"'),
    ]:
        col  = CORAL if 'CONV' in label else GREEN
        bg   = '#FFF0EE' if 'CONV' in label else '#EDFAF3'
        ax.add_patch(mpatches.FancyBboxPatch(
            (x, 0.69), 0.43, 0.13, boxstyle='round,pad=0.01',
            facecolor=bg, edgecolor=col, linewidth=1.5,
            transform=ax.transAxes, clip_on=False))
        ax.text(x+0.215, 0.77, label, ha='center', fontsize=9.5,
                fontweight='bold', color=col, transform=ax.transAxes)
        ax.text(x+0.215, 0.73, example, ha='center', fontsize=8.5,
                color=DGRAY, transform=ax.transAxes, style='italic')

    ax.annotate('', xy=(0.52, 0.755), xytext=(0.48, 0.755),
                xycoords='axes fraction', textcoords='axes fraction',
                arrowprops=dict(arrowstyle='->', color=NAVY, lw=2))

    # ── Architecture flow
    ax.text(0.01, 0.67, 'System Architecture', fontsize=10, fontweight='bold', color=NAVY)
    ax.axhline(0.65, color=NAVY, linewidth=1.0, xmin=0.01, xmax=0.99)

    boxes = [
        (0.01, 0.38, 0.18, 0.24, 'DATA LAYER',
         ['KOSIS Life Table 2024', 'BOK ECOS (CPI / rates)', 'KRX (KOSPI / KTB)', 'MDIS Household Survey'],
         '#D6EAF8', TEAL),
        (0.22, 0.38, 0.28, 0.24, 'STATISTICAL MODELS',
         ['Survival Analysis (Cox PH)', 'Monte Carlo 10,000 paths',
          'CVaR Portfolio (extensible: 5+ assets)', 'Pension NPV Optimizer',
          'K-Means Clustering (K=5)', 'Anomaly Detection (3 layers)'],
         '#D5F5E3', GREEN),
        (0.53, 0.38, 0.20, 0.24, 'AI AGENT LAYER',
         ['Claude API Tool Use', 'Multi-turn context', 'Plain-language output', 'Elderly UX adaptation'],
         '#FDEBD0', GOLD),
        (0.76, 0.38, 0.22, 0.24, 'SLOW BANKING UI',
         ['Streamlit (large font)', 'Plotly fan charts', 'Voice I/O (Whisper+gTTS)', 'Guardian alerts'],
         '#F5EEF8', PURP),
    ]
    for bx, by, bw, bh, btitle, blines, bbg, btc in boxes:
        ax.add_patch(mpatches.FancyBboxPatch(
            (bx, by), bw, bh, boxstyle='round,pad=0.01',
            facecolor=bbg, edgecolor=btc, linewidth=1.2,
            transform=ax.transAxes, clip_on=False))
        ax.text(bx+bw/2, by+bh-0.015, btitle, ha='center',
                fontsize=8, fontweight='bold', color=btc, transform=ax.transAxes)
        for i, ln in enumerate(blines):
            ax.text(bx+0.01, by+bh-0.058-i*0.033, f'  {ln}',
                    fontsize=7.8, color=DGRAY, transform=ax.transAxes)
    for x1, x2 in [(0.19, 0.22), (0.50, 0.53), (0.73, 0.76)]:
        ax.annotate('', xy=(x2, 0.50), xytext=(x1, 0.50),
                    xycoords='axes fraction', textcoords='axes fraction',
                    arrowprops=dict(arrowstyle='->', color=NAVY, lw=1.8))

    # ── Data sources table
    ax.text(0.01, 0.36, 'Data Sources', fontsize=10, fontweight='bold', color=NAVY)
    ax.axhline(0.34, color=NAVY, linewidth=1.0, xmin=0.01, xmax=0.99)

    headers  = ['Dataset', 'Source', 'Period', 'Usage']
    rows_src = [
        ['National Life Table 2024', 'Statistics Korea (KOSIS)', '2024', 'Survival Analysis'],
        ['CPI Monthly Index',        'Bank of Korea ECOS API',   '2000-2026', 'Inflation distribution'],
        ['Market Interest Rates',    'Bank of Korea ECOS',       '2000-2026', 'Bond yield proxy'],
        ['KOSPI Daily Price',        'KRX',                      '2000-2026', 'Equity return dist.'],
        ['KTB Bond Total Return',    'KRX (18 files)',           '2009-2026', 'Bond return dist.'],
        ['Household Finance Survey', 'MDIS (Statistics Korea)',  '2025',      'K-Means clustering'],
    ]
    col_x = [0.01, 0.28, 0.52, 0.68]
    col_w = [0.27, 0.24, 0.16, 0.31]
    y0 = 0.32
    for ci, h in enumerate(headers):
        ax.add_patch(mpatches.Rectangle(
            (col_x[ci], y0-0.03), col_w[ci], 0.03,
            facecolor=NAVY, transform=ax.transAxes, clip_on=False))
        ax.text(col_x[ci]+0.005, y0-0.005, h, fontsize=8, fontweight='bold',
                color='white', va='center', transform=ax.transAxes)
    for ri, row in enumerate(rows_src):
        bg = LGRAY if ri % 2 == 0 else 'white'
        for ci, cell in enumerate(row):
            ax.add_patch(mpatches.Rectangle(
                (col_x[ci], y0-0.03*(ri+2)), col_w[ci], 0.03,
                facecolor=bg, edgecolor='#DDDDDD', linewidth=0.5,
                transform=ax.transAxes, clip_on=False))
            ax.text(col_x[ci]+0.005, y0-0.03*(ri+1)-0.005, cell,
                    fontsize=7.8, va='center', color=DGRAY, transform=ax.transAxes)

    pdf.savefig(fig, bbox_inches='tight')
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2: SURVIVAL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
def page_survival(pdf):
    fig = plt.figure(figsize=(11, 8.5))
    add_header(fig, 'Survival Analysis — Statistical Methodology',
               'Life Table + Cox Proportional Hazards Model')
    gs = GridSpec(2, 2, figure=fig, top=0.90, bottom=0.06,
                  left=0.06, right=0.97, hspace=0.42, wspace=0.32)

    # ── (0,0) Life table
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.axis('off')
    ax1.set_title('2024 National Life Table (Male)', fontsize=10,
                  fontweight='bold', color=NAVY, pad=6)
    tdata = [
        ['Age', 'q(x)  death prob.', 'l(x)  survivors', 'S(x)  surv. rate'],
        ['0',   '0.00261', '100,000', '1.0000'],
        ['40',  '0.00123',  '98,125', '0.9813'],
        ['60',  '0.00645',  '94,199', '0.9420'],
        ['75',  '0.02152',  '73,840', '0.7384'],
        ['80',  '0.04870',  '64,373', '0.6437'],
        ['90',  '0.13441',  '27,541', '0.2754'],
        ['100+','1.00000',   '1,153', '0.0115'],
    ]
    tbl = ax1.table(cellText=tdata[1:], colLabels=tdata[0],
                    cellLoc='center', loc='center', bbox=[0, 0.03, 1, 0.92])
    tbl.auto_set_font_size(False); tbl.set_fontsize(8)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor(NAVY)
            cell.set_text_props(color='white', fontweight='bold')
        elif r % 2 == 0:
            cell.set_facecolor(LGRAY)
        cell.set_edgecolor('#DDDDDD')
    ax1.text(0.5, -0.03, 'Source: Statistics Korea, 2024 Complete Life Table',
             ha='center', fontsize=7.5, color='gray', transform=ax1.transAxes)

    # ── (0,1) Cox PH steps
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis('off')
    ax2.set_title('Cox Proportional Hazards — Implementation', fontsize=10,
                  fontweight='bold', color=NAVY, pad=6)
    steps = [
        ('Step 1  Baseline Survival', '#D6EAF8',
         r'$S_0(t) = \prod_{x=0}^{t-1}(1 - q_x)$' + '\n'
         'Cumulative product of life-table death probs.\n'
         'Male/Female tables applied separately.'),
        ('Step 2  Conditional on Current Age', '#D5F5E3',
         r'$S(t\,|\,a) = S_0(t)\,/\,S_0(a)$' + '\n'
         'Normalizes survival to current age a.\n'
         'Eliminates left-truncation bias.'),
        ('Step 3  Cox PH Adjustment', '#FDEBD0',
         r'$S_{adj}(t) = [S(t|a)]^{HR_{total}}$' + '\n'
         r'$HR_{total} = HR_{smoke} \times HR_{chronic} \times HR_{BMI}$' + '\n'
         'Smoking 1.50 / Chronic 1.35 / Obese 1.25 / Underweight 1.40'),
        ('Step 4  Death Age Sampling', '#F5EEF8',
         r'$P(T=t)=S(t)-S(t+1)$  (discrete hazard)' + '\n'
         'Inverse-transform sampling from P(T=t).\n'
         'Feeds directly into Monte Carlo paths.'),
    ]
    y = 0.97
    for title, bg, desc in steps:
        h = 0.22
        ax2.add_patch(mpatches.FancyBboxPatch(
            (0, y-h), 1, h, boxstyle='round,pad=0.01',
            facecolor=bg, edgecolor='#CCCCCC', linewidth=0.7,
            transform=ax2.transAxes))
        ax2.text(0.02, y-0.015, title, fontsize=8.5, fontweight='bold',
                 color=NAVY, va='top', transform=ax2.transAxes)
        ax2.text(0.03, y-0.065, desc, fontsize=8, color=DGRAY,
                 va='top', transform=ax2.transAxes)
        y -= (h + 0.015)

    # ── (1,0) Survival curves
    ax3 = fig.add_subplot(gs[1, 0])
    profiles = [
        (PersonalProfile('M', 60, False, False, 22.0),
         'Male healthy (HR=1.00)', TEAL, '-'),
        (PersonalProfile('M', 60, True,  True,  28.0),
         'Male smoke+chronic (HR=2.03)', CORAL, '--'),
        (PersonalProfile('F', 60, False, False, 22.0),
         'Female healthy (HR=1.00)', GOLD, '-'),
    ]
    for prof, label, color, ls in profiles:
        sf = survival.get_survival_function(prof)
        exp = survival.expected_remaining_life(prof)
        ax3.plot(sf.index, sf.values * 100, color=color, linestyle=ls,
                 linewidth=2, label=f'{label}  (E[T]={60+exp:.1f}y)')
    ax3.axvline(83, color='gray', ls=':', lw=1, alpha=0.7)
    ax3.text(83.4, 6, '83y', fontsize=8, color='gray')
    ax3.set_xlabel('Age (years)'); ax3.set_ylabel('Survival Probability (%)')
    ax3.set_title('Personalized Survival Functions S(t)', fontweight='bold', color=NAVY)
    ax3.legend(loc='upper right', fontsize=7.5)
    ax3.set_xlim(60, 100); ax3.set_ylim(0, 106)
    ax3.grid(True, alpha=0.3); ax3.set_facecolor(LGRAY)

    # ── (1,1) Death age distribution + note on mean vs median
    ax4 = fig.add_subplot(gs[1, 1])
    for prof, label, color, ls in profiles:
        samp = survival.sample_death_age(prof, 10_000)
        ax4.hist(samp, bins=np.arange(60, 102), alpha=0.50,
                 color=color, density=True, label=label)
        med = np.median(samp)
        mn  = np.mean(samp)
        ax4.axvline(med, color=color, ls='--', lw=1.5, alpha=0.9)
        ax4.axvline(mn,  color=color, ls=':',  lw=1.2, alpha=0.7)
    ax4.set_xlabel('Death Age (years)'); ax4.set_ylabel('Density')
    ax4.set_title('Simulated Death Age Distribution (10,000 paths)',
                  fontweight='bold', color=NAVY)
    ax4.set_xlim(60, 102)
    ax4.grid(True, alpha=0.3); ax4.set_facecolor(LGRAY)
    ax4.text(0.97, 0.95,
             'dashed = median  |  dotted = mean\n'
             'Median (~85) > Mean (83.2)\n'
             r'$\Rightarrow$ LEFT-skewed' + '\n'
             'Hazard h(t) accelerates in late life\n'
             '-> deaths cluster at mode (~88)\n'
             '-> thin left tail (low h at 60-75)\n'
             '-> hard bound at ~100 compresses right',
             transform=ax4.transAxes, ha='right', va='top', fontsize=7.2,
             color='gray', style='italic',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.88))

    pdf.savefig(fig, bbox_inches='tight')
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3: MONTE CARLO METHODOLOGY
# ══════════════════════════════════════════════════════════════════════════════
def page_mc(pdf):
    fig = plt.figure(figsize=(11, 8.5))
    add_header(fig, 'Monte Carlo Simulation — Statistical Methodology',
               'Multivariate Return Sampling  +  Age-Varying Medical Shocks')
    gs = GridSpec(2, 2, figure=fig, top=0.90, bottom=0.06,
                  left=0.06, right=0.97, hspace=0.42, wspace=0.32)

    # ── (0,0) Distribution table
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.axis('off')
    ax1.set_title('Historical Distribution Parameters  (KOSPI 2000-2025 / KTB 2009-2025)', fontsize=10,
                  fontweight='bold', color=NAVY, pad=6)
    tdata = [
        ['Variable', 'Distribution', 'Mean', 'Std Dev'],
        ['KOSPI equity', 'Normal', f'{params.stock_mean:.2%}', f'{params.stock_std:.2%}'],
        ['KTB bond',     'Normal', f'{params.bond_mean:.2%}',  f'{params.bond_std:.2%}'],
        ['CPI inflation','Normal (clipped)', f'{params.infl_mean:.2%}', f'{params.infl_std:.2%}'],
        ['Medical shock','Pois x LogNorm', 'age-varying', '500 +/- 300 (10k KRW)'],
        ['Equity-Bond corr.', 'Multivariate Normal', f'{params.corr:.3f}', '—'],
    ]
    tbl = ax1.table(cellText=tdata[1:], colLabels=tdata[0],
                    cellLoc='center', loc='center', bbox=[0, 0.03, 1, 0.90])
    tbl.auto_set_font_size(False); tbl.set_fontsize(8)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor(NAVY)
            cell.set_text_props(color='white', fontweight='bold')
        elif r % 2 == 0:
            cell.set_facecolor(LGRAY)
        cell.set_edgecolor('#DDDDDD')
    ax1.text(0.5, -0.03, 'Incomplete year (2026) excluded from parameter estimation.',
             ha='center', fontsize=7.5, color='gray', transform=ax1.transAxes)

    # ── (0,1) Algorithm
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis('off')
    ax2.set_title('Simulation Algorithm', fontsize=10,
                  fontweight='bold', color=NAVY, pad=6)
    algo = (
        'FOR i = 1 ... N  (N = 10,000 paths)\n'
        '  T_i ~ SurvivalModel.sample_death_age()\n\n'
        '  FOR t = 0 ... T_i - 1\n'
        '    age = current_age + t\n\n'
        '    [r_equity, r_bond] ~ MVN([u_s, u_b], Sigma)\n'
        '    r_port = w * r_equity + (1-w) * r_bond\n\n'
        '    pi_t ~ Normal(u_pi, s_pi) clipped [-2%, 15%]\n\n'
        '    lambda(age): <70->0.05  70-79->0.10  80+->0.20\n'
        '    n_t ~ Poisson(lambda(age))\n'
        '    M_t = n_t * LogNormal(mu_m, sigma_m)\n\n'
        '    cum_infl = prod(1 + pi_s, s=0..t)\n'
        '    E_t = E_0 * cum_infl          [real expense]\n'
        '    P_t = pension * cum_infl      [CPI-indexed]\n'
        '    D_t = E_t + M_t - P_t        [net drawdown]\n\n'
        '    A_{t+1} = A_t * (1 + r_port) - D_t\n'
        '    IF A_{t+1} <= 0: record depletion age, stop\n\n'
        'P(depletion) = #{depleted paths} / N\n'
        'Percentile paths: empirical quantile per age\n\n'
        '[Current] inflation sampled i.i.d. per year.\n'
        'Future: AR(1)/VAR(1) for CPI-rate persistence\n'
        '(captures inflation clustering across years).'
    )
    ax2.text(0.02, 0.97, algo, transform=ax2.transAxes, fontsize=7.6,
             va='top', family='monospace', color=DGRAY,
             bbox=dict(boxstyle='round', facecolor=LGRAY,
                       edgecolor='#CCCCCC', linewidth=0.8))

    # ── (1,0) Annual return distributions
    ax3 = fig.add_subplot(gs[1, 0])
    x_range = np.linspace(-0.9, 1.2, 300)
    ax3.hist(stock_a * 100, bins=15, alpha=0.55, color=CORAL, density=True,
             label=f'KOSPI annual  (mean={params.stock_mean:.1%})')
    ax3.hist(bond_a * 100,  bins=15, alpha=0.55, color=TEAL,  density=True,
             label=f'KTB annual   (mean={params.bond_mean:.1%})')
    for vals, color in [(stock_a, CORAL), (bond_a, TEAL)]:
        mu_fit, sd_fit = vals.mean(), vals.std()
        ax3.plot(x_range * 100, norm.pdf(x_range, mu_fit, sd_fit),
                 color=color, lw=1.5, ls='--', alpha=0.85)
    ax3.axvline(0, color='black', lw=0.8, ls='--')
    ax3.set_xlabel('Annual Return (%)'); ax3.set_ylabel('Density')
    ax3.set_title('Historical Annual Returns  (bars) vs Normal Fit (dashed)',
                  fontweight='bold', color=NAVY)
    ax3.legend(); ax3.grid(True, alpha=0.3); ax3.set_facecolor(LGRAY)
    ax3.set_xlim(-80, 120)

    # ── (1,1) Inflation time series
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.fill_between(infl_ts.index, infl_ts.values, alpha=0.35, color=GOLD)
    ax4.plot(infl_ts.index, infl_ts.values, color=GOLD, lw=1.2)
    ax4.axhline(params.infl_mean * 100, color=CORAL, ls='--', lw=1.5,
                label=f'Mean {params.infl_mean:.2%}')
    ax4.axhline(0, color='gray', lw=0.7)
    ax4.fill_between(infl_ts.index,
                     (params.infl_mean - params.infl_std) * 100,
                     (params.infl_mean + params.infl_std) * 100,
                     alpha=0.15, color=CORAL, label='mean +/- 1 SD')
    ax4.set_xlabel('Year'); ax4.set_ylabel('CPI YoY Change (%)')
    ax4.set_title('Korea CPI Inflation Time Series  (2001-2025)',
                  fontweight='bold', color=NAVY)
    ax4.legend(); ax4.grid(True, alpha=0.3); ax4.set_facecolor(LGRAY)

    pdf.savefig(fig, bbox_inches='tight')
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4: STATISTICAL ROBUSTNESS
# ══════════════════════════════════════════════════════════════════════════════
def page_robustness(pdf):
    fig = plt.figure(figsize=(11, 8.5))
    add_header(fig, 'Statistical Robustness & Model Validation',
               'Normality Defense  |  Age-Varying Lambda  |  Sensitivity Analysis')
    gs = GridSpec(2, 2, figure=fig, top=0.90, bottom=0.06,
                  left=0.06, right=0.97, hspace=0.45, wspace=0.35)

    # ── (0,0) Normality: monthly vs annual Q-Q
    ax1 = fig.add_subplot(gs[0, 0])
    monthly_vals = stock_m.values
    annual_vals  = stock_a.values
    (osm_m, osr_m), (slope_m, intercept_m, r_m) = stats.probplot(monthly_vals, dist='norm')
    (osm_a, osr_a), (slope_a, intercept_a, r_a) = stats.probplot(annual_vals,  dist='norm')
    ax1.scatter(osm_m, osr_m, s=8,  color=CORAL, alpha=0.5, label='Monthly returns')
    ax1.scatter(osm_a, osr_a, s=25, color=TEAL,  alpha=0.9, label='Annual returns', zorder=5)
    lx = np.array([osm_m.min(), osm_m.max()])
    ax1.plot(lx, slope_m*lx + intercept_m, color=CORAL, ls='--', lw=1)
    ax1.plot(lx, slope_a*lx + intercept_a, color=TEAL,  ls='--', lw=1.5)
    ax1.set_xlabel('Theoretical Quantiles'); ax1.set_ylabel('Sample Quantiles')
    ax1.set_title('Q-Q Plot: Monthly vs Annual KOSPI Returns\n(closer to diagonal = more normal)',
                  fontweight='bold', color=NAVY)
    ax1.legend(); ax1.grid(True, alpha=0.3); ax1.set_facecolor(LGRAY)

    # ── (0,1) Normality test table
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis('off')
    ax2.set_title('Normality Tests — Monthly vs Annual', fontsize=10,
                  fontweight='bold', color=NAVY, pad=6)

    monthly_kurt = float(stats.kurtosis(monthly_vals, fisher=True))
    monthly_skew = float(stats.skew(monthly_vals))
    _, sw_m_p    = stats.shapiro(monthly_vals[:50])  # Shapiro max 5000
    _, jb_m_p    = stats.jarque_bera(monthly_vals)
    _, jb_a_p    = stats.jarque_bera(annual_vals)

    rows = [
        ['Metric', 'Monthly', 'Annual', 'Threshold'],
        ['N observations', f'{len(monthly_vals)}', f'{len(annual_vals)}', '—'],
        ['Mean', f'{monthly_vals.mean():.3%}', f'{annual_vals.mean():.3%}', '—'],
        ['Std Dev', f'{monthly_vals.std():.3%}', f'{annual_vals.std():.3%}', '—'],
        ['Excess Kurtosis', f'{monthly_kurt:.2f}', f'{stock_kurt:.2f}', '0.0 (normal)'],
        ['Skewness', f'{monthly_skew:.2f}', f'{stock_skew:.2f}', '0.0 (normal)'],
        ['Shapiro-Wilk p', f'{sw_m_p:.3f}', f'{sw_p:.3f}', '> 0.05 = normal'],
        ['Jarque-Bera p', f'{jb_m_p:.3f}', f'{jb_a_p:.3f}', '> 0.05 = normal'],
        ['Normality', 'REJECTED', 'NOT REJECTED', '—'],
    ]
    tbl = ax2.table(cellText=rows[1:], colLabels=rows[0],
                    cellLoc='center', loc='center', bbox=[0, 0.02, 1, 0.94])
    tbl.auto_set_font_size(False); tbl.set_fontsize(8)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor(NAVY)
            cell.set_text_props(color='white', fontweight='bold')
        else:
            row_data = rows[r]
            if r == len(rows) - 1:
                if c == 1: cell.set_facecolor('#FADBD8')
                elif c == 2: cell.set_facecolor('#D5F5E3')
            elif r % 2 == 0:
                cell.set_facecolor(LGRAY)
        cell.set_edgecolor('#DDDDDD')
    ax2.text(0.5, -0.02,
             'Simulation uses ANNUAL returns -> normality holds by CLT.',
             ha='center', fontsize=8, color=GREEN, fontweight='bold',
             transform=ax2.transAxes)

    # ── (1,0) Age-varying lambda: depletion prob comparison
    ax3 = fig.add_subplot(gs[1, 0])
    ages_plot = np.arange(60, 96)
    lam_old   = np.full_like(ages_plot, 0.08, dtype=float)
    lam_new   = np.where(ages_plot < 70, 0.05, np.where(ages_plot < 80, 0.10, 0.20))
    ax3.step(ages_plot, lam_old, color=CORAL, lw=2, ls='--', label='Old: constant lambda=0.08')
    ax3.step(ages_plot, lam_new, color=GREEN, lw=2.5, label='New: age-stratified lambda')
    ax3.fill_between(ages_plot, lam_old, lam_new, where=lam_new > lam_old,
                     alpha=0.15, color=GREEN, label='Additional risk (80+)')
    ax3.fill_between(ages_plot, lam_old, lam_new, where=lam_new < lam_old,
                     alpha=0.15, color=CORAL, label='Reduced risk (<70)')
    ax3.set_xlabel('Age'); ax3.set_ylabel('Medical Shock Rate (events/year)')
    ax3.set_title('Medical Cost Lambda: Before vs After Fix\n'
                  '(NHIS: medical spending ~2x per decade after 65)',
                  fontweight='bold', color=NAVY)
    ax3.legend(fontsize=7.5); ax3.grid(True, alpha=0.3); ax3.set_facecolor(LGRAY)
    ax3.text(0.97, 0.90,
             f'Depletion prob delta:\nCase A:  {OLD_A.depletion_prob:.1%} -> {RES_A.depletion_prob:.1%}\n'
             f'Change: +{(RES_A.depletion_prob - OLD_A.depletion_prob)*100:.1f}pp',
             transform=ax3.transAxes, ha='right', va='top', fontsize=8,
             color=NAVY, bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

    # ── (1,1) Sensitivity tornado chart
    ax4 = fig.add_subplot(gs[1, 1])
    labels = list(SENS.keys())
    deltas = [(SENS[k] - BASE_DP) * 100 for k in labels]
    colors = [CORAL if d > 0 else GREEN for d in deltas]
    y_pos  = np.arange(len(labels))
    ax4.barh(y_pos, deltas, color=colors, alpha=0.80, edgecolor='white', height=0.65)
    ax4.axvline(0, color='black', lw=1)
    ax4.set_yticks(y_pos); ax4.set_yticklabels(labels, fontsize=8)
    ax4.set_xlabel('Change in Depletion Probability (pp)')
    ax4.set_title(f'Sensitivity Analysis\n(Base depletion: {BASE_DP:.1%}  |  Case A, 1,500 paths)',
                  fontweight='bold', color=NAVY)
    ax4.grid(True, axis='x', alpha=0.3); ax4.set_facecolor(LGRAY)
    for i, d in enumerate(deltas):
        ax4.text(d + (0.3 if d >= 0 else -0.3), i,
                 f'{d:+.1f}pp', va='center', ha='left' if d >= 0 else 'right',
                 fontsize=7.5, color=DGRAY)

    pdf.savefig(fig, bbox_inches='tight')
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5: SEQUENCE-OF-RETURNS RISK + K-SELECTION
# ══════════════════════════════════════════════════════════════════════════════
def page_sequence_and_k(pdf):
    fig = plt.figure(figsize=(11, 8.5))
    add_header(fig, 'Model Validation — Sequencing Risk & Cluster Selection',
               'Sequence-of-Returns Quantification  |  Silhouette-Based K Verification')
    gs = GridSpec(2, 2, figure=fig, top=0.90, bottom=0.06,
                  left=0.06, right=0.97, hspace=0.48, wspace=0.35)

    # ── (0,0) Sequence-of-returns bar chart ──────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    labels   = SEQ_RISK['quartile'].tolist()
    dep_vals = (SEQ_RISK['depletion_prob'] * 100).tolist()
    avg_ret  = (SEQ_RISK['avg_early_return'] * 100).tolist()
    bar_colors = [CORAL, '#E8965A', '#82C87A', GREEN]
    bars = ax1.bar(range(4), dep_vals, color=bar_colors, alpha=0.85,
                   edgecolor='white', linewidth=1.2, width=0.65)
    for i, (v, r) in enumerate(zip(dep_vals, avg_ret)):
        ax1.text(i, v + 0.8, f'{v:.1f}%', ha='center', fontsize=9,
                 fontweight='bold', color=DGRAY)
        ax1.text(i, v / 2, f'avg return\n{r:+.1f}%', ha='center',
                 fontsize=7.5, color='white', fontweight='bold')
    ax1.set_xticks(range(4))
    ax1.set_xticklabels([l.replace(' (', '\n(') for l in labels], fontsize=8)
    ax1.set_ylabel('Depletion Probability (%)')
    ax1.set_title('Sequence-of-Returns Risk  (Case A)\n'
                  'Paths stratified by first-5-year avg portfolio return',
                  fontweight='bold', color=NAVY)
    ax1.set_ylim(0, 100); ax1.grid(True, axis='y', alpha=0.3)
    ax1.set_facecolor(LGRAY)
    q1p, q4p = dep_vals[0], dep_vals[3]
    ax1.annotate('', xy=(3, q4p + 2), xytext=(0, q1p + 2),
                 arrowprops=dict(arrowstyle='<->', color=NAVY, lw=1.5))
    ax1.text(1.5, max(q1p, q4p) + 5,
             f'{q1p - q4p:.1f}pp gap',
             ha='center', fontsize=9, color=NAVY, fontweight='bold')

    # ── (0,1) Explanation ────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis('off')
    ax2.set_title('Why Sequencing Risk Matters', fontsize=10,
                  fontweight='bold', color=NAVY, pad=6)

    explanation = [
        ('What it shows',
         'All four quartiles are drawn from the SAME underlying return\n'
         'distribution (Normal, mean=9.6%, sd=28.0%).\n'
         'Only the ORDER of returns differs across paths.'),
        ('The mechanism',
         'Early bad returns reduce the capital base before recovery.\n'
         'With lower assets, subsequent withdrawals consume a higher\n'
         'FRACTION of remaining wealth — accelerating depletion.\n'
         'Conversely, strong early returns build a buffer that absorbs\n'
         'later downturns.'),
        ('Implication for our model',
         'Monte Carlo captures this effect implicitly: every path\n'
         'independently samples its own return sequence. The full\n'
         'depletion probability (e.g., 65%) is the expectation over\n'
         'ALL possible sequences, including adversely ordered ones.\n'
         'The fan chart width reflects sequencing uncertainty.'),
        ('Key finding',
         f'Q1 paths (avg early return {avg_ret[0]:+.1f}%): {q1p:.1f}% depletion\n'
         f'Q4 paths (avg early return {avg_ret[3]:+.1f}%): {q4p:.1f}% depletion\n'
         f'Gap = {q1p - q4p:.1f}pp — sequence dominates long-run outcome.'),
    ]
    y = 0.96
    for title, text in explanation:
        bg = '#EEF2F8'
        h  = 0.19 + text.count('\n') * 0.025
        ax2.add_patch(mpatches.FancyBboxPatch(
            (0, y - h), 1, h, boxstyle='round,pad=0.01',
            facecolor=bg, edgecolor='#BBBBBB', linewidth=0.7,
            transform=ax2.transAxes, clip_on=False))
        ax2.text(0.012, y - 0.013, title, transform=ax2.transAxes,
                 fontsize=8.5, fontweight='bold', color=NAVY, va='top')
        ax2.text(0.018, y - 0.052, text, transform=ax2.transAxes,
                 fontsize=7.8, color=DGRAY, va='top')
        y -= (h + 0.018)

    # ── (1,0) K-selection silhouette plot ────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    from models.clustering import HouseholdClusterModel
    k_vals = sorted(HouseholdClusterModel.K_SILHOUETTE.keys())
    s_vals = [HouseholdClusterModel.K_SILHOUETTE[k] for k in k_vals]
    opt_k  = HouseholdClusterModel.OPTIMAL_K
    bar_c  = [GREEN if k == opt_k else TEAL for k in k_vals]
    ax3.bar(k_vals, s_vals, color=bar_c, alpha=0.80,
            edgecolor='white', linewidth=1.0, width=0.65)
    for k, s in zip(k_vals, s_vals):
        ax3.text(k, s + 0.002, f'{s:.4f}', ha='center', fontsize=7.8,
                 fontweight='bold' if k == opt_k else 'normal',
                 color=GREEN if k == opt_k else DGRAY)
    ax3.set_xlabel('Number of Clusters K')
    ax3.set_ylabel('Silhouette Score')
    ax3.set_title('K-Means Cluster Selection  (K = 2 … 9)\n'
                  'Silhouette analysis on 18,521 households (5,000-sample approx.)',
                  fontweight='bold', color=NAVY)
    ax3.set_xticks(k_vals)
    ax3.set_ylim(0.26, 0.40)
    ax3.grid(True, axis='y', alpha=0.3); ax3.set_facecolor(LGRAY)
    ax3.text(opt_k, s_vals[k_vals.index(opt_k)] + 0.008,
             f'  K={opt_k} optimal', fontsize=9, color=GREEN, fontweight='bold')

    # ── (1,1) K-selection interpretation ─────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    ax4.set_title('K-Selection: Results & Interpretation', fontsize=10,
                  fontweight='bold', color=NAVY, pad=6)

    interp_rows = [
        ['K', 'Silhouette', 'Interpretation'],
        ['2', '0.3675', 'Splits only rich vs. rest — too coarse'],
        ['3', '0.3591', 'Adds middle class; misses asset tiers'],
        ['4', '0.3651', 'Close to K=5 but conflates top tiers'],
        ['5 *', '0.3746', 'Optimal — separates 5 wealth strata clearly'],
        ['6', '0.3165', 'Over-segments; ultra-rich cluster fragments'],
        ['7', '0.3154', 'Further fragmentation, lower cohesion'],
        ['8', '0.3133', 'Diminishing returns on cluster granularity'],
        ['9', '0.2870', 'Excessive — many near-empty clusters'],
    ]
    tbl = ax4.table(cellText=interp_rows[1:], colLabels=interp_rows[0],
                    cellLoc='left', loc='center', bbox=[0, 0.0, 1, 0.94])
    tbl.auto_set_font_size(False); tbl.set_fontsize(8)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor(NAVY)
            cell.set_text_props(color='white', fontweight='bold')
        elif r == 4:   # K=5 row
            cell.set_facecolor('#D5F5E3')
            cell.set_text_props(fontweight='bold', color=GREEN)
        elif r % 2 == 0:
            cell.set_facecolor(LGRAY)
        cell.set_edgecolor('#DDDDDD')

    pdf.savefig(fig, bbox_inches='tight')
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6: SIMULATION RESULTS
# ══════════════════════════════════════════════════════════════════════════════
def page_results(pdf):  # noqa: F811
    fig = plt.figure(figsize=(11, 8.5))
    add_header(fig, 'Monte Carlo Simulation Results',
               '4,000-Path Simulation  |  Age-Varying Medical Lambda Applied')
    gs = GridSpec(2, 2, figure=fig, top=0.90, bottom=0.06,
                  left=0.06, right=0.97, hspace=0.42, wspace=0.32)

    cases = [
        ('Case A: Male 60, Assets 300M KRW\nExpense 250K/mo, Pension 100K/mo, w=0.40',
         CASE_A, RES_A, CORAL),
        ('Case B: Female 60, Assets 500M KRW\nExpense 300K/mo, Pension 150K/mo, w=0.30',
         CASE_B, RES_B, TEAL),
    ]

    for idx, (title, si, res, color) in enumerate(cases):
        ax = fig.add_subplot(gs[0, idx])
        pp   = res.percentile_paths
        ages = pp.index
        ax.fill_between(ages, pp['p10']/10_000, pp['p90']/10_000,
                        alpha=0.18, color=color, label='10th-90th pct. band')
        ax.fill_between(ages, pp['p10']/10_000, pp['p50']/10_000,
                        alpha=0.10, color=color)
        ax.plot(ages, pp['p50']/10_000, color=color, lw=2.2, label='Median (p50)')
        ax.plot(ages, pp['p10']/10_000, color=color, lw=1, ls='--', alpha=0.7, label='p10')
        ax.plot(ages, pp['p90']/10_000, color=color, lw=1, ls=':', alpha=0.7, label='p90')
        ax.axhline(0, color='gray', lw=0.8)
        ax.set_xlabel('Age (years)'); ax.set_ylabel('Assets (100M KRW)')
        ax.set_title(title, fontsize=8.8, fontweight='bold', color=NAVY)
        ax.legend(fontsize=7.5); ax.grid(True, alpha=0.3); ax.set_facecolor(LGRAY)
        ax.text(0.97, 0.95,
                f'Depletion Prob: {res.depletion_prob:.1%}\n'
                f'Median at death: {res.median_final_assets/10_000:.1f}B KRW',
                transform=ax.transAxes, ha='right', va='top', fontsize=8.5,
                color=color, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
        note = ('Note: p90 initial rise reflects high-return scenarios\n'
                'where investment gains exceed net drawdown (correct behavior)')
        ax.text(0.02, 0.02, note, transform=ax.transAxes, fontsize=7,
                color='gray', style='italic', va='bottom')

    # ── (1,0) Depletion age distributions
    ax3 = fig.add_subplot(gs[1, 0])
    for title, si, res, color in cases:
        if len(res.depletion_age_dist) > 0:
            lbl = title.split('\n')[0]
            ax3.hist(res.depletion_age_dist, bins=np.arange(60, 102),
                     alpha=0.55, color=color, density=True,
                     label=f'{lbl}  ({res.depletion_prob:.1%} depleted)')
            ax3.axvline(np.median(res.depletion_age_dist), color=color,
                        ls='--', lw=1.8)
    ax3.set_xlabel('Age at Depletion'); ax3.set_ylabel('Density (among depleted paths)')
    ax3.set_title('Depletion Age Distribution\n(dashed = median depletion age)',
                  fontweight='bold', color=NAVY)
    ax3.legend(fontsize=7.5); ax3.grid(True, alpha=0.3); ax3.set_facecolor(LGRAY)

    # ── (1,1) Percentile table
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    ax4.set_title('Asset Path Summary Table  (100M KRW)', fontsize=10,
                  fontweight='bold', color=NAVY, pad=6)
    hdr  = ['Age', 'Case A p10', 'Case A p50', 'Case A p90',
                   'Case B p10', 'Case B p50', 'Case B p90']
    rows = []
    for age in [65, 70, 75, 80, 85, 90]:
        row = [f'{age}y']
        for res in [RES_A, RES_B]:
            pp = res.percentile_paths
            if age in pp.index:
                for col in ['p10', 'p50', 'p90']:
                    row.append(f'{pp.loc[age, col]/10_000:.2f}')
            else:
                row += ['—', '—', '—']
        rows.append(row)
    rows.append([
        'Depletion',
        f'{RES_A.depletion_prob:.1%}', '—', '—',
        f'{RES_B.depletion_prob:.1%}', '—', '—',
    ])
    tbl = ax4.table(cellText=rows, colLabels=hdr,
                    cellLoc='center', loc='center', bbox=[0, 0.0, 1, 0.96])
    tbl.auto_set_font_size(False); tbl.set_fontsize(8)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor(NAVY)
            cell.set_text_props(color='white', fontweight='bold')
        elif r == len(rows):
            cell.set_facecolor('#FEF9E7')
        elif r % 2 == 0:
            cell.set_facecolor(LGRAY)
        if c in (1, 2, 3): cell.set_facecolor(cell.get_facecolor())
        cell.set_edgecolor('#DDDDDD')

    pdf.savefig(fig, bbox_inches='tight')
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6: ASSUMPTIONS, LIMITATIONS & REBUTTALS
# ══════════════════════════════════════════════════════════════════════════════
def page_limitations(pdf):
    fig = plt.figure(figsize=(11, 8.5))
    add_header(fig, 'Statistical Assumptions, Limitations & Rebuttals',
               'Transparent disclosure strengthens — not weakens — analytical credibility')
    ax = fig.add_axes([0.02, 0.01, 0.96, 0.91])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')

    sections = [
        {
            'title': 'A  Cox PH Independence Assumption  (HR multiplication)',
            'bg': '#EAF2FF', 'tc': NAVY,
            'rows': [
                ('Assumption',  'HR_total = HR_smoke x HR_chronic x HR_BMI  implies full independence between covariates.'),
                ('Concern',     'Smoking and chronic disease are positively correlated. '
                                'Joint HR may be overestimated for multi-risk profiles.'),
                ('Rebuttal',    'Published HRs sourced from epidemiological meta-analyses that control for '
                                'confounders via multivariable regression.  The individual HRs are themselves '
                                'adjusted estimates, partially accounting for covariate correlation.'),
                ('Residual risk', 'Residual confounding remains.  Estimated overestimation: ~10-20% for '
                                  'smoke+chronic combined profile (HR=2.03 vs plausible ~1.70-1.85).'),
                ('Mitigation',  'Ideally: fit Cox PH from individual-level censored survival data (NHIS microdata). '
                                'Unavailable for this project.  Adding interaction term H_smoke*H_chronic would require '
                                'empirical joint HR estimates from Korean cohort studies.'),
            ]
        },
        {
            'title': 'B  Population Life Table vs Individual-Level Survival Data',
            'bg': '#EDFAF3', 'tc': GREEN,
            'rows': [
                ('Assumption',  'Baseline hazard S_0(t) derived from aggregate national life table, '
                                'not from individual censored survival records.'),
                ('Justification', 'NHIS individual-level survival microdata with censoring indicators is not '
                                  'publicly available.  Life table provides the only viable baseline.'),
                ('Limitation',  'All individuals share the same baseline hazard function S_0(t). '
                                'Heterogeneity within age/sex strata (socioeconomic, genetic) is unmodeled.'),
                ('Impact',      'Cox HR adjustment partially individualizes the curve.  '
                                'Residual heterogeneity causes predictions to regress toward population mean.'),
                ('Note on median vs mean', 'E[T] from life table = 23.2y => mean death age 83.2y. '
                                           'Simulated median ~85y gives Mode(~88) > Median(85) > Mean(83.2): LEFT-skewed. '
                                           'Mechanism: the baseline hazard h(t) accelerates sharply in late life '
                                           '(Gompertz-like), concentrating deaths near the modal age (~88). '
                                           'In the early conditional interval (age 60-75), h(t) remains low — '
                                           'few deaths occur here, but they extend the distribution leftward, '
                                           'forming a long thin left tail relative to the compressed right side. '
                                           'The hard upper bound at ~100 further truncates the right tail. '
                                           'Both effects together produce left-skewness: '
                                           'Mean < Median, consistent with our simulation. '
                                           '(Right-skewed would imply Mean > Median — contradicted by data.)'),
            ]
        },
        {
            'title': 'C  Normal Distribution for Equity Returns',
            'bg': '#FFFBEA', 'tc': '#A07800',
            'rows': [
                ('Concern raised', 'KOSPI monthly returns exhibit fat tails.  Normal distribution may '
                                   'underestimate tail-loss probability.'),
                ('Empirical rebuttal', f'Simulation operates at ANNUAL frequency.  Annual KOSPI returns: '
                                       f'excess kurtosis = {stock_kurt:.2f} (normal = 0.0), '
                                       f'skewness = {stock_skew:.2f}, '
                                       f'Shapiro-Wilk p = {sw_p:.2f} >> 0.05.  '
                                       'Normality NOT rejected at annual level.'),
                ('CLT argument',   'Aggregating 12 monthly returns to annual via geometric product reduces '
                                   'kurtosis substantially (central limit theorem).  '
                                   'Monthly excess kurtosis = {:.2f} collapses to {:.2f} annually.'.format(
                                       float(stats.kurtosis(stock_m.values, fisher=True)), stock_kurt)),
                ('Residual concern', 'Rare extreme years (e.g., -46.5% in 2000) remain in the empirical distribution. '
                                     'Normal fit still samples plausible but bounded extreme values.'),
                ('Alternative considered', 't-distribution fit yields df=35.1 — effectively indistinguishable '
                                           'from Normal at this sample size.  Historical simulation would '
                                           'require IID assumption on annual blocks.'),
            ]
        },
        {
            'title': 'D  Medical Cost Shock Model  (lambda fix applied in this version)',
            'bg': '#EDFAF3', 'tc': GREEN,
            'rows': [
                ('Previous version', 'Constant lambda = 0.08/year for all ages — '
                                     'one major shock every ~12.5 years regardless of age.'),
                ('Problem',         '80-year-olds face the same modeled shock frequency as 65-year-olds.  '
                                    'NHIS data: per-capita medical spending roughly doubles each decade after 65.'),
                ('Current version',  'Age-stratified lambda:  age < 70 -> 0.05 (1 per 20y),  '
                                    '70-79 -> 0.10 (1 per 10y),  80+ -> 0.20 (1 per 5y).'),
                ('Impact measured',  f'Case A depletion probability: {OLD_A.depletion_prob:.1%} (old) '
                                    f'-> {RES_A.depletion_prob:.1%} (new).  '
                                    f'Delta = +{(RES_A.depletion_prob-OLD_A.depletion_prob)*100:.1f}pp.  '
                                    'More conservative and empirically grounded.'),
                ('Remaining limitation', 'Shock magnitude (mean 500, SD 300, 10k KRW) is constant across age.  '
                                         'In reality, catastrophic late-life care costs can exceed 10M KRW/event.  '
                                         'Future: age-varying LogNormal parameters.'),
            ]
        },
        {
            'title': 'E  Correlation Stationarity  &  Portfolio Constraint',
            'bg': '#F5EEF8', 'tc': PURP,
            'rows': [
                ('Assumption',      f'KOSPI-KTB correlation treated as constant at {params.corr:.3f} (2009-2025).'),
                ('Known limitation', 'Correlation regimes shift.  In inflationary periods (2022-2023), '
                                     'equity-bond correlation turned positive in many markets.'),
                ('Sensitivity',     'Sensitivity analysis shows stock ratio change of +/-0.2 moves depletion '
                                    f'probability by {abs(SENS["Stock ratio +0.2"]-BASE_DP)*100:.1f} / '
                                    f'{abs(SENS["Stock ratio -0.2"]-BASE_DP)*100:.1f} pp — moderate impact.'),
                ('2-asset limitation', 'CVaR optimization with only KOSPI + KTB is mathematically degenerate: '
                                     'the feasible set is a single line segment w in [0,1], '
                                     'so the "efficient frontier" collapses to a scalar weight. '
                                     'A simple target-return solve would yield identical results. '
                                     'We do NOT claim this as a core differentiator of the current system.'),
                ('Actual role',     'The CVaR module serves as a principled risk-profiling layer: '
                                    'user risk tolerance (conservative/moderate/aggressive) '
                                    '-> optimal KOSPI weight -> Monte Carlo stock_ratio. '
                                    'This integration chain is functionally correct and non-trivial for the UX. '
                                    'CVaR provides downside-risk framing that variance-based methods miss.'),
                ('Future value',    'With 5+ assets (domestic equity, foreign equity, REITs, IG bonds, alternatives), '
                                    'CVaR optimization becomes genuinely non-trivial: '
                                    'the feasible set is an N-1 dimensional simplex and Rockafellar-Uryasev '
                                    'linear reformulation provides significant computational advantage over MVO. '
                                    'Architecture is already extensible — only DataLoader expansion required.'),
                ('AR(1) plan',      'CPI and KTB yields exhibit positive serial correlation (AR(1) structure visible in time series plot, p.3). '
                                    'Current model samples inflation i.i.d. each year — understates persistence in high-inflation regimes. '
                                    'Future: VAR(1) joint model for CPI + yield dynamics. '
                                    'Expected effect: wider fan chart spread and higher depletion prob under sustained inflation scenarios.'),
            ]
        },
        {
            'title': 'F  Anomaly Detection Evaluation — Synthetic Test Data',
            'bg': '#FEF9E7', 'tc': '#A07800',
            'rows': [
                ('Validation setup',  'F1-score (0.952) and AUC-ROC (1.000) reported for Layer 1 '
                                      '(transaction anomaly detection) were computed on SYNTHETIC test data, '
                                      'not on real labeled fraud records. '
                                      'This must be stated explicitly to avoid misleading evaluation.'),
                ('Why synthetic',     'Real Korean financial fraud labels (ground-truth tagged transactions) '
                                      'are not publicly available. No open dataset with Korean banking '
                                      'transaction fraud labels exists for academic/competition use. '
                                      'Synthetic cases were designed to reflect known fraud patterns '
                                      '(voice phishing: new account + nighttime + consecutive + large amount).'),
                ('Test set design',   'Normal: 30 small-amount daytime payments (3-40 10k KRW). '
                                      'Boundary: 10 moderate transfers to existing accounts (50-120 10k KRW). '
                                      'Anomaly-high: 20 large transfers (300-800 10k KRW). '
                                      'Anomaly-phishing: 20 nighttime new-account consecutive transfers. '
                                      'Boundary cases deliberately included to avoid inflated metrics.'),
                ('Honest interpretation', 'The high AUC reflects that the engineered features (new account flag, '
                                          'hour-of-day, consecutive count) are highly discriminative for the '
                                          'synthetic patterns tested. Real-world performance against '
                                          'sophisticated fraud will be lower. '
                                          'Proper validation requires labeled transaction data from a financial institution.'),
                ('Layer 3 caveat',    'Life pattern detection (Layer 3) uses entirely synthetic daily activity '
                                      'signals. No real elderly monitoring data was available. '
                                      'Threshold values (card transactions, conversation counts) are '
                                      'assumptions, not empirically calibrated.'),
            ]
        },
    ]

    y_cursor = 0.985
    lh = 0.013
    sec_pad = 0.014

    for sec in sections:
        # Count total text lines
        n_lines = sum(1 + len(v) // 90 + v.count('\n') for _, v in sec['rows'])
        box_h   = 0.040 + n_lines * lh + len(sec['rows']) * 0.002

        ax.add_patch(mpatches.FancyBboxPatch(
            (0, y_cursor - box_h), 1.0, box_h,
            boxstyle='round,pad=0.005',
            facecolor=sec['bg'], edgecolor=sec['tc'], linewidth=1.1,
            transform=ax.transAxes, clip_on=False))
        ax.text(0.008, y_cursor - 0.008, sec['title'], transform=ax.transAxes,
                fontsize=9, fontweight='bold', color=sec['tc'], va='top')
        ty = y_cursor - 0.032
        for key, val in sec['rows']:
            line = f'{key}:  {val}'
            ax.text(0.015, ty, line, transform=ax.transAxes,
                    fontsize=7.8, color=DGRAY, va='top', wrap=True,
                    bbox=None)
            n = 1 + len(line) // 100
            ty -= lh * (n + 0.3)

        y_cursor -= (box_h + sec_pad)

    pdf.savefig(fig, bbox_inches='tight')
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 8: AGENT CONVERSATION DEMO
# ══════════════════════════════════════════════════════════════════════════════
def page_agent_demo(pdf):
    fig = plt.figure(figsize=(11, 8.5))
    add_header(fig, 'AI Agent Layer — Statistical Output to Elderly Natural Language',
               'Tool Use Pipeline  |  Slow Banking UX  |  False Positive Handling')

    ax = fig.add_axes([0.01, 0.01, 0.98, 0.91])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')

    KO_FONT = {'fontproperties': matplotlib.font_manager.FontProperties(family='Noto Sans KR')}

    def _text(x, y, s, **kwargs):
        has_ko = any('가' <= c <= '힣' for c in s)
        if has_ko:
            fp = matplotlib.font_manager.FontProperties(
                family='Noto Sans KR',
                size=kwargs.pop('fontsize', 7.7))
            ax.text(x, y, s, fontproperties=fp,
                    transform=ax.transAxes, **kwargs)
        else:
            ax.text(x, y, s, transform=ax.transAxes, **kwargs)

    def cbox2(x, y, w, h, title, lines, bg='#EEF2F8', tc=NAVY, fs=7.7, spacing=0.033):
        ax.add_patch(mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle='round,pad=0.01',
            facecolor=bg, edgecolor=tc, linewidth=1.1,
            transform=ax.transAxes, clip_on=False))
        ax.text(x+0.010, y+h-0.012, title, transform=ax.transAxes,
                fontsize=8.5, fontweight='bold', color=tc, va='top')
        for i, ln in enumerate(lines):
            _text(x+0.014, y+h-0.048-i*spacing, ln,
                  fontsize=fs, color=DGRAY, va='top')

    _text(0.50, 0.977,
          'Core Design: Bridging Probabilistic Rigor and Human Understanding',
          ha='center', fontsize=10.5, fontweight='bold', color=NAVY, va='top')
    ax.axhline(0.958, color=NAVY, linewidth=1.3, xmin=0.01, xmax=0.99)

    # ── Top row: A → B → C pipeline (y = 0.47 to 0.95)
    TOP, MID = 0.95, 0.47

    # A: Statistical Engine Output
    cbox2(0.01, MID, 0.295, TOP - MID,
          'A  STATISTICAL ENGINE OUTPUT',
          ['Profile:  Age 72  |  Male  |  Chronic dx',
           'Assets: 200M KRW | Expense: 220K/mo',
           'Pension: 100K/mo | Non-smoker',
           '',
           'Tool: run_monte_carlo()',
           '  P(depletion < 85y) =  0.68',
           '  median_depletion_age  = 81',
           '  median_final_assets   = 85M KRW',
           '',
           'Tool: run_clustering()',
           '  cluster = 3  (mid-wealth elderly)',
           '  peer_wealth_percentile = 34'],
          bg='#D6EAF8', tc=TEAL)

    ax.annotate('', xy=(0.330, 0.715), xytext=(0.305, 0.715),
                xycoords='axes fraction', textcoords='axes fraction',
                arrowprops=dict(arrowstyle='->', color=NAVY, lw=2.2))

    # B: Agent Transformation
    cbox2(0.33, MID, 0.295, TOP - MID,
          'B  AGENT TRANSFORMATION',
          ['Age Segment: 3  (>= 75 years old)',
           'Prompt: warm + concise + 1 core message',
           '',
           'Language Rules Applied:',
           '  0.68  ->  "열 번 중 일곱 번"',
           '  81y   ->  "여든한 살 즈음"',
           '  34th  ->  "100명 중 34번째"',
           '',
           'Tool-call logic:',
           '  "얼마나 버티나요?"',
           '  --> run_monte_carlo(profile)',
           '  Parameters carried from context'],
          bg='#FDEBD0', tc=GOLD)

    ax.annotate('', xy=(0.660, 0.715), xytext=(0.625, 0.715),
                xycoords='axes fraction', textcoords='axes fraction',
                arrowprops=dict(arrowstyle='->', color=NAVY, lw=2.2))

    # C: Agent Output
    cbox2(0.66, MID, 0.33, TOP - MID,
          'C  AGENT OUTPUT  (Slow Banking UI)',
          ['User (age 72):',
           '"제 돈이 얼마나 버틸까요?"',
           '',
           'Agent:',
           '"어르신, 컴퓨터가 계산을 해보니',
           ' 지금처럼 생활하시면 여든한 살',
           ' 즈음에는 주머니 사정이 조금',
           ' 가벼워지실 수 있어요.',
           '',
           ' 열 번 중에 일곱 번 정도는',
           ' 돈이 먼저 마를 수 있다고 해요.',
           '',
           ' 걱정 마세요. 연금을 더 유리하게',
           ' 타는 방법을 알려드릴게요."'],
          bg='#EDFAF3', tc=GREEN, spacing=0.031)

    # ── Divider
    ax.axhline(0.455, color='#AAAAAA', linewidth=0.9, xmin=0.01, xmax=0.99)
    _text(0.50, 0.445, 'False Positive Handling  &  Multi-Turn Context Continuity',
          ha='center', fontsize=9, fontweight='bold', color=NAVY, va='top')

    # ── Bottom row (y = 0.01 to 0.42)
    BOT = 0.40

    # D: Anomaly FP Handling
    cbox2(0.01, 0.01, 0.46, BOT,
          'D  ANOMALY DETECTION: FALSE POSITIVE HANDLING',
          ['Trigger: Z-score 2.31 | New account | 23:15',
           '350K KRW  (3x personal baseline) | score: 0.847',
           '',
           'Design: NO abrupt block.',
           'Elderly confusion > transaction security.',
           'False positive cost << false negative cost.',
           '',
           'Agent (Slow Banking confirmation flow):',
           '"어르신, 잠깐만요! 평소 안 보내시던',
           ' 계좌에 큰 금액이에요.',
           ' 본인이 직접 하신 거 맞나요?"',
           '',
           '[YES]  ->  proceed + 24h review flag',
           '[NO / hesitate / 30s no-response]',
           '  -> Hold + guardian alert + branch call'],
          bg='#FEF9E7', tc='#A07800', spacing=0.028)

    # E: Multi-turn context
    cbox2(0.50, 0.01, 0.49, BOT,
          'E  MULTI-TURN CONTEXT CONTINUITY',
          ['Scenario: User refines conditions iteratively',
           '',
           'Turn 1: "지출을 200만원으로 줄이면요?"',
           '  expense: 250K -> 200K  (rest unchanged)',
           '  Depletion:  68% -> 55%  (-13pp)',
           '',
           'Turn 2: "담배도 끊으면요?"',
           '  smoking: True->False | HR: 2.03->1.35',
           '  E[death]: 83.2y -> 84.8y',
           '  Depletion: 55% -> 49%  (-6pp)',
           '',
           'Turn 3: "주식을 줄이면 더 안전해요?"',
           '  run_portfolio(conservative) -> KOSPI 12%',
           '  Depletion: 49% -> 46%  (-3pp)',
           'Agent never re-asks confirmed parameters.'],
          bg='#F5EEF8', tc=PURP, spacing=0.028)

    pdf.savefig(fig, bbox_inches='tight')
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    output = 'LifeLong_WM_Analysis_Report.pdf'
    print('Generating PDF report...')
    with PdfPages(output) as pdf:
        page_overview(pdf);         print('  Page 1/8  Project Overview')
        page_survival(pdf);         print('  Page 2/8  Survival Analysis')
        page_mc(pdf);               print('  Page 3/8  Monte Carlo Methodology')
        page_robustness(pdf);       print('  Page 4/8  Statistical Robustness')
        page_sequence_and_k(pdf);   print('  Page 5/8  Sequencing Risk & K Selection')
        page_results(pdf);          print('  Page 6/8  Simulation Results')
        page_limitations(pdf);      print('  Page 7/8  Assumptions & Limitations')
        page_agent_demo(pdf);       print('  Page 8/8  Agent Conversation Demo')
        d = pdf.infodict()
        d['Title']  = 'LifeLong WM AI Agent — Quantitative Analysis Report'
        d['Author'] = 'JB Financial Group Fin:AI Challenge 2026'
    print(f'\nSaved: {output}')
