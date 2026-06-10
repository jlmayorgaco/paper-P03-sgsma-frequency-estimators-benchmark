# Estimator reference support

Date: 2026-06-10

Purpose: keep every estimator module tied to at least one real paper. The
machine-readable source is `src/estimators/references.py`; each estimator module
declares its own `REFERENCE_KEYS` near the top of the file. This file is the
human audit table.

The references below are method-family support unless the note says otherwise.
They are not a claim that the implementation is a byte-for-byte reproduction of
the cited paper. Experimental estimators are especially conservative: several
currently inherit the shared experimental compatibility base and should not be
reported as validated implementations until promoted through dedicated tests.

## Active estimators

| Module | Label | Reference keys | Support note |
| --- | --- | --- | --- |
| `zcd.py` | ZCD | `djuric2008_zero_crossing` | Power-network frequency measurement using zero-crossing/Fourier techniques. |
| `ipdft.py` | IPDFT | `grandke1983_ipdft` | Foundational interpolated DFT support. |
| `tft.py` | TFT | `platasgarza2010_tft` | Taylor/maximally-flat differentiator support for dynamic phasor and frequency estimates. |
| `rls.py` | RLS | `carlsson1994_rls_notch` | RLS/notch-model support for adaptive sinusoidal frequency tracking. |
| `pll.py` | PLL | `kaura1997_pll_distorted` | Grid PLL operation under distorted utility conditions. |
| `sogi_pll.py` | SOGI-PLL | `ciobotaru2006_sogi_pll` | Single-phase SOGI-PLL structure. |
| `sogi_fll.py` | SOGI-FLL | `ciobotaru2006_sogi_pll`, `rodriguez2011_multiresonant_fll` | SOGI/PLL structure plus FLL grid-synchronization support. |
| `type3_sogi_pll.py` | Type-3 SOGI-PLL | `kaura1997_pll_distorted`, `ciobotaru2006_sogi_pll` | Internal benchmark variant derived from PLL and SOGI-PLL families; not a claim of an exact type-3 paper reproduction. |
| `lkf.py` | LKF | `kalman1960_linear_filtering`, `pradhan2004_complex_lkf` | Linear Kalman filtering plus complex LKF phasor-estimation support. |
| `lkf2.py` | LKF2 | `kalman1960_linear_filtering`, `reza2012_frequency_adaptive_lkf` | Frequency-adaptive LKF support for grid voltage parameters. |
| `ekf.py` | EKF | `kalman1960_linear_filtering`, `dash1999_extended_complex_kalman` | EKF/extended-complex Kalman support for distorted power-system frequency estimation. |
| `ukf.py` | UKF | `julier2004_unscented_filtering`, `regulski2012_ukf_frequency` | Unscented filtering theory plus grid-frequency estimation support. |
| `ra_ekf.py` | RA-EKF | `dash1999_extended_complex_kalman`, `panigrahi2009_robust_extended_kalman` | Robust/adaptive EKF-family support for distorted power-system signals. |
| `tkeo.py` | TKEO | `maragos1993_energy_separation` | Teager-Kaiser energy separation support for AM/FM demodulation. |
| `prony.py` | Prony | `hauer1991_prony_power_system` | Prony analysis support for modal content in power-system response. |
| `esprit.py` | ESPRIT | `roy1989_esprit` | Foundational ESPRIT signal-parameter estimation support. |
| `koopman.py` | Koopman (RK-DPMU) | `williams2015_edmd_koopman` | EDMD/Koopman operator approximation support. |
| `pi_gru.py` | PI-GRU | `cho2014_gru`, `raissi2019_pinn` | GRU architecture plus physics-informed learning support; checkpoint-specific claims still require weight hash disclosure. |

## Experimental and extra estimators

| Module | Label | Reference keys | Support note |
| --- | --- | --- | --- |
| `ckf.py` | CKF | `arasaratnam2009_cubature_kalman` | Cubature Kalman filter method-family support. |
| `sr_ukf.py` | SR-UKF | `julier2004_unscented_filtering`, `vandermerwe2001_srukf` | Unscented filtering plus square-root UKF support. |
| `adaptive_ekf.py` | Adaptive-EKF | `kalman1960_linear_filtering`, `mehra1970_adaptive_kalman` | Adaptive Kalman variance-estimation support. |
| `imm_ekf_ukf.py` | IMM-EKF/UKF | `julier2004_unscented_filtering`, `blom1988_interacting_multiple_model` | IMM switching-model support plus UKF-family support. |
| `hinf_frequency_kf.py` | Hinf-KF | `kalman1960_linear_filtering`, `shaked1992_hinf_estimation` | H-infinity state-estimation support. |
| `wls_ipdft.py` | WLS-IpDFT | `grandke1983_ipdft`, `belega2008_weighted_ipdft` | Interpolated DFT plus weighted multipoint IpDFT support. |
| `quinn_fernandes.py` | Quinn-Fernandes | `quinn1991_frequency_estimation` | Foundational Quinn-Fernandes frequency estimator. |
| `jacobsen_interpolated_dft.py` | Jacobsen-Interpolated-DFT | `grandke1983_ipdft`, `jacobsen2007_frequency_estimators` | DFT interpolation and fast frequency-estimator support. |
| `sliding_least_squares.py` | Sliding-Least-Squares | `besson1999_nonlinear_least_squares` | Least-squares sinusoidal frequency-estimation support. |
| `music.py` | MUSIC | `schmidt1986_music` | Foundational MUSIC signal-parameter estimation support. |
| `matrix_pencil.py` | Matrix-Pencil | `hua1990_matrix_pencil` | Matrix pencil sinusoid-parameter estimation support. |
| `hilbert_phase_derivative.py` | Hilbert-Phase-Derivative | `boashash1992_instantaneous_frequency` | Instantaneous-frequency/Hilbert-phase support. |
| `epll.py` | EPLL | `karimi2004_epll_sync` | Enhanced PLL support for polluted and variable-frequency grid environments. |
| `music_experimental.py` | MUSIC | `schmidt1986_music` | Legacy experimental wrapper; same method-family support as `music.py`. |

## Reference key index

| Key | Paper |
| --- | --- |
| `arasaratnam2009_cubature_kalman` | Arasaratnam and Haykin, "Cubature Kalman Filters," IEEE TAC, 2009, DOI: 10.1109/TAC.2009.2019800. |
| `belega2008_weighted_ipdft` | Belega and Dallet, "Frequency estimation via weighted multipoint interpolated DFT," IET SMT, 2008, DOI: 10.1049/IET-SMT:20070022. |
| `besson1999_nonlinear_least_squares` | Besson and Stoica, "Nonlinear Least-Squares Approach to Frequency Estimation and Detection for Sinusoidal Signals with Arbitrary Envelope," Digital Signal Processing, 1999, DOI: 10.1006/DSPR.1998.0330. |
| `blom1988_interacting_multiple_model` | Blom and Bar-Shalom, "The interacting multiple model algorithm for systems with Markovian switching coefficients," IEEE TAC, 1988, DOI: 10.1109/9.1299. |
| `boashash1992_instantaneous_frequency` | Boashash, "Estimating and interpreting the instantaneous frequency of a signal. I. Fundamentals," Proceedings of the IEEE, 1992, DOI: 10.1109/5.135376. |
| `carlsson1994_rls_notch` | Carlsson and Handel, "A notch filter based on recursive least-squares modelling," Signal Processing, 1994, DOI: 10.1016/0165-1684(94)90213-5. |
| `cho2014_gru` | Cho et al., "Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation," EMNLP, 2014. |
| `ciobotaru2006_sogi_pll` | Ciobotaru, Teodorescu, and Blaabjerg, "A new single-phase PLL structure based on second order generalized integrator," PESC, 2006, DOI: 10.1109/PESC.2006.1711988. |
| `dash1999_extended_complex_kalman` | Dash, Pradhan, and Panda, "Frequency estimation of distorted power system signals using extended complex Kalman filter," IEEE TPWRD, 1999, DOI: 10.1109/61.772312. |
| `djuric2008_zero_crossing` | Djuric and Djurisic, "Frequency measurement of distorted signals using Fourier and zero crossing techniques," EPSR, 2008, DOI: 10.1016/j.epsr.2008.01.008. |
| `grandke1983_ipdft` | Grandke, "Interpolation Algorithms for Discrete Fourier Transforms of Weighted Signals," IEEE TIM, 1983, DOI: 10.1109/TIM.1983.4315077. |
| `hauer1991_prony_power_system` | Hauer, "Application of Prony analysis to the determination of modal content and equivalent models for measured power system response," IEEE TPS, 1991, DOI: 10.1109/59.119247. |
| `hua1990_matrix_pencil` | Hua and Sarkar, "Matrix pencil method for estimating parameters of exponentially damped/undamped sinusoids in noise," IEEE TASSP, 1990, DOI: 10.1109/29.56027. |
| `jacobsen2007_frequency_estimators` | Jacobsen and Kootsookos, "Fast, Accurate Frequency Estimators," IEEE SPM, 2007, DOI: 10.1109/MSP.2007.361611. |
| `julier2004_unscented_filtering` | Julier and Uhlmann, "Unscented Filtering and Nonlinear Estimation," Proceedings of the IEEE, 2004, DOI: 10.1109/JPROC.2003.823141. |
| `kalman1960_linear_filtering` | Kalman, "A New Approach to Linear Filtering and Prediction Problems," Journal of Basic Engineering, 1960, DOI: 10.1115/1.3662552. |
| `karimi2004_epll_sync` | Karimi-Ghartemani and Iravani, "A Method for Synchronization of Power Electronic Converters in Polluted and Variable-Frequency Environments," IEEE TPS, 2004, DOI: 10.1109/TPWRS.2004.831280. |
| `kaura1997_pll_distorted` | Kaura and Blasko, "Operation of a phase locked loop system under distorted utility conditions," IEEE TIA, 1997, DOI: 10.1109/28.567077. |
| `maragos1993_energy_separation` | Maragos, Kaiser, and Quatieri, "Energy separation in signal modulations with application to speech analysis," IEEE TSP, 1993, DOI: 10.1109/78.277799. |
| `mehra1970_adaptive_kalman` | Mehra, "On the identification of variances and adaptive Kalman filtering," IEEE TAC, 1970, DOI: 10.1109/TAC.1970.1099422. |
| `panigrahi2009_robust_extended_kalman` | Panigrahi, Rauta, and Panda, "Robust extended complex Kalman Filter applied to distorted power system signals for frequency estimation," ICPS, 2009, DOI: 10.1109/ICPWS.2009.5442752. |
| `platasgarza2010_tft` | Platas-Garza and de la O Serna, "Dynamic Phasor and Frequency Estimates Through Maximally Flat Differentiators," IEEE TIM, 2010, DOI: 10.1109/TIM.2009.2030921. |
| `pradhan2004_complex_lkf` | Pradhan, "Voltage phasor estimation using complex linear Kalman filter," DPSP, 2004, DOI: 10.1049/CP:20040054. |
| `quinn1991_frequency_estimation` | Quinn and Fernandes, "A fast efficient technique for the estimation of frequency," Biometrika, 1991, DOI: 10.1093/BIOMET/78.3.489. |
| `raissi2019_pinn` | Raissi, Perdikaris, and Karniadakis, "Physics-informed neural networks," Journal of Computational Physics, 2019, DOI: 10.1016/J.JCP.2018.10.045. |
| `regulski2012_ukf_frequency` | Regulski and Terzija, "Estimation of Frequency and Fundamental Power Components Using an Unscented Kalman Filter," IEEE TIM, 2012, DOI: 10.1109/TIM.2011.2179342. |
| `reza2012_frequency_adaptive_lkf` | Reza, Ciobotaru, and Agelidis, "Frequency adaptive linear Kalman filter for fast and accurate estimation of grid voltage parameters," POWERCON, 2012, DOI: 10.1109/POWERCON.2012.6401446. |
| `rodriguez2011_multiresonant_fll` | Rodriguez et al., "Multiresonant Frequency-Locked Loop for Grid Synchronization of Power Converters Under Distorted Grid Conditions," IEEE TIE, 2011, DOI: 10.1109/TIE.2010.2042420. |
| `roy1989_esprit` | Roy and Kailath, "ESPRIT-estimation of signal parameters via rotational invariance techniques," IEEE TASSP, 1989, DOI: 10.1109/29.32276. |
| `schmidt1986_music` | Schmidt, "Multiple emitter location and signal parameter estimation," IEEE TAP, 1986, DOI: 10.1109/TAP.1986.1143830. |
| `shaked1992_hinf_estimation` | Shaked and Theodor, "A frequency domain approach to the problems of H-infinity-minimum error state estimation and deconvolution," IEEE TSP, 1992, DOI: 10.1109/78.175743. |
| `vandermerwe2001_srukf` | Van der Merwe and Wan, "The square-root unscented Kalman filter for state and parameter-estimation," ICASSP, 2001, DOI: 10.1109/ICASSP.2001.940586. |
| `williams2015_edmd_koopman` | Williams, Kevrekidis, and Rowley, "A Data-Driven Approximation of the Koopman Operator: Extending Dynamic Mode Decomposition," Journal of Nonlinear Science, 2015, DOI: 10.1007/S00332-015-9258-5. |
