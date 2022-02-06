%OSEA Detects and classifies QRS complexes in ECGs
%   skeleton = OSEA(ecg)
%
%   Input:
%   -ecg:           input ECG signal (must be samples at 200Hz)
%
%   Output:
%   -skeleton:      vector of the same dimensions as "ecg" which contains a
%                   QRS type code at each sample location
%
%   Possible type codes:
%   	NOTQRS	0	/* not-QRS (not a getann/putann code) */
%       NORMAL	1	/* normal beat */
%       LBBB	2	/* left bundle branch block beat */
%       RBBB	3	/* right bundle branch block beat */
%       ABERR	4	/* aberrated atrial premature beat */
%       PVC     5	/* premature ventricular contraction */
%       FUSION	6	/* fusion of ventricular and normal beat */
%       NPC     7	/* nodal (junctional) premature beat */
%       APC     8	/* atrial premature contraction */
%       SVPB	9	/* premature or ectopic supraventricular beat */
%       VESC	10	/* ventricular escape beat */
%       NESC	11	/* nodal (junctional) escape beat */
%       PACE	12	/* paced beat */
%       UNKNOWN	13	/* unclassifiable beat */
%       NOISE	14	/* signal quality change */
%       ARFCT	16	/* isolated QRS-like artifact */
%       STCH	18	/* ST change */
%       TCH     19	/* T-wave change */
%       SYSTOLE	20	/* systole */
%       DIASTOLE 21	/* diastole */
%       NOTE	22	/* comment annotation */
%       MEASURE 23	/* measurement annotation */
%       BBB     25	/* left or right bundle branch block */
%       PACESP	26	/* non-conducted pacer spike */
%       RHYTHM	28	/* rhythm change */
%       LEARN	30	/* learning */
%       FLWAV	31	/* ventricular flutter wave */
%       VFON	32	/* start of ventricular flutter/fibrillation */
%       VFOFF	33	/* end of ventricular flutter/fibrillation */
%       AESC	34	/* atrial escape beat */
%       SVESC	35	/* supraventricular escape beat */
%       NAPC	37	/* non-conducted P-wave (blocked APB) */
%       PFUS	38	/* fusion of paced and normal beat */
%       PQ      39	/* PQ junction (beginning of QRS) */
%       JPT     40	/* J point (end of QRS) */
%       RONT	41	/* R-on-T premature ventricular contraction */
%
