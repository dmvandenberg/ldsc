#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""munge_sumstats.py (Python-2.7 compatible)

Extra features vs. upstream LDSC (https://github.com/bulik/ldsc)
--------------------------------
* **--prioritize-hm** — if two or more candidate columns tie for the “most
  filled”, pick an `HM_*` column when available.

* **Global HM flag** — once the signed‑summary‑statistic picker ends up
  selecting an `HM_*` column, the module‑level constant `HM_PRIORITY_ACTIVE` 
  is set to **True**.  Any later calls to the generic `column_picker()` 
  (e.g. when resolving Allele, MAF, or RSID ties) will *implicitly* prefer an 
  `HM_*` column whenever ties occur, even if `--prioritize-hm` was not passed 
  for those specific picks.

* NA drop exception was added for INFO and FRQ columns: Originally, if you would
  want FRQ in you output, only rows with a value would be included, sometimes
  resulting in great loss.

* **--keep-maf** is superseded if all values in MAF column are NA

* Added some logic to deal with 'non-numeric' P-values

"""

from __future__ import division

import time
import os
import sys
import traceback
import gzip
import bz2
import argparse

import pandas as pd
import numpy as np
from scipy.stats import chi2

from ldscore import sumstats
from ldsc import MASTHEAD, Logger, sec_to_str

# -----------------------------------------------------------------------------
# Global configuration
# -----------------------------------------------------------------------------

np.seterr(invalid='ignore')

try:
    x = pd.DataFrame({'A': [1, 2, 3]})
    x.sort_values(by='A')
except AttributeError:
    raise ImportError('LDSC requires pandas version >= 0.17.0')

HM_PRIORITY_ACTIVE = False # becomes True once an hm_* column is selected

# -----------------------------------------------------------------------------
# Constant dictionaries
# -----------------------------------------------------------------------------

null_values = {
    'LOG_ODDS': 0,
    'BETA': 0,
    'OR': 1,
    'Z': 0
}

default_cnames = {
    # RS NUMBER
    'SNP': 'SNP',
    'MARKERNAME': 'SNP',
    'SNPID': 'SNP',
    'RS': 'SNP',
    'RSID': 'SNP',
    'RS_NUMBER': 'SNP',
    'RS_NUMBERS': 'SNP',
    # CHROMOSOME
    'CHR': 'CHR',
    'CHROM': 'CHR',
    # BASE PAIR LOCATION
    'BP': 'BP',
    'POS': 'BP',
    'BPOS': 'BP',
    # NUMBER OF STUDIES
    'NSTUDY': 'NSTUDY',
    'N_STUDY': 'NSTUDY',
    'NSTUDIES': 'NSTUDY',
    'N_STUDIES': 'NSTUDY',
    # P-VALUE
    'P': 'P',
    'PVALUE': 'P',
    'P_VALUE':  'P',
    'P-VALUE':  'P',
    'PVAL': 'P',
    'P_VAL': 'P',
    'GC_PVALUE': 'P',
    # ALLELE 1
    'A1': 'A1',
    'ALLELE1': 'A1',
    'ALLELE_1': 'A1',
    'EFFECT_ALLELE': 'A1',
    'REFERENCE_ALLELE': 'A1',
    'INC_ALLELE': 'A1',
    'EA': 'A1',
    # ALLELE 2
    'A2': 'A2',
    'ALLELE2': 'A2',
    'ALLELE_2': 'A2',
    'OTHER_ALLELE': 'A2',
    'NON_EFFECT_ALLELE': 'A2',
    'DEC_ALLELE': 'A2',
    'NEA': 'A2',
    # N
    'N': 'N',
    'NCASE': 'N_CAS',
    'CASES_N': 'N_CAS',
    'N_CASE': 'N_CAS',
    'N_CASES': 'N_CAS',
    'N_CONTROLS': 'N_CON',
    'N_CAS': 'N_CAS',
    'N_CON': 'N_CON',
    'N_CASE': 'N_CAS',
    'NCONTROL': 'N_CON',
    'CONTROLS_N': 'N_CON',
    'N_CONTROL': 'N_CON',
    'WEIGHT': 'N',  # metal does this. possibly risky.
    # SIGNED STATISTICS
    'ZSCORE': 'Z',
    'Z-SCORE': 'Z',
    'GC_ZSCORE': 'Z',
    'Z': 'Z',
    'OR': 'OR',
    'B': 'BETA',
    'BETA': 'BETA',
    'LOG_ODDS': 'LOG_ODDS',
    'EFFECTS': 'BETA',
    'EFFECT': 'BETA',
    'SIGNED_SUMSTAT': 'SIGNED_SUMSTAT',
    'ODDS_RATIO': 'OR',
    # INFO
    'INFO': 'INFO',
    # MAF
    'EAF': 'FRQ',
    'FRQ': 'FRQ',
    'MAF': 'FRQ',
    'FRQ_U': 'FRQ',
    'F_U': 'FRQ',
    # HM COLUMNS (GWAS Catalog)
    'HM_RSID': 'SNP',
    'HM_CHROM': 'CHR',
    'HM_POS': 'BP',
    'HM_EFFECT_ALLELE': 'A1',
    'HM_OTHER_ALLELE': 'A2',
    'HM_BETA': 'BETA',
    'HM_ODDS_RATIO': 'OR',
    'HM_EFFECT_ALLELE_FREQUENCY': 'FRQ',
}

describe_cname = {
    'SNP': 'Variant ID (e.g., rs number)',
    'CHR': 'Chromosome position',
    'BP': 'Basepair location',
    'P': 'p-Value',
    'A1': 'Allele 1, interpreted as ref allele for signed sumstat.',
    'A2': 'Allele 2, interpreted as non-ref allele for signed sumstat.',
    'EAF': 'Allele frequency of A1',
    'N': 'Sample size',
    'N_CAS': 'Number of cases',
    'N_CON': 'Number of controls',
    'Z': 'Z-score (0 --> no effect; above 0 --> A1 is trait/risk increasing)',
    'OR': 'Odds ratio (1 --> no effect; above 1 --> A1 is risk increasing)',
    'BETA': '[linear/logistic] regression coefficient (0 --> no effect; above 0 --> A1 is trait/risk increasing)',
    'LOG_ODDS': 'Log odds ratio (0 --> no effect; above 0 --> A1 is risk increasing)',
    'INFO': 'INFO score (imputation quality; higher --> better imputation)',
    'FRQ': 'Allele frequency',
    'SIGNED_SUMSTAT': 'Directional summary statistic as specified by --signed-sumstats.',
    'NSTUDY': 'Number of studies in which the SNP was genotyped.'
}

numeric_cols = ['P', 
                'N', 
                'N_CAS', 
                'N_CON', 
                'Z', 
                'OR', 
                'BETA', 
                'LOG_ODDS', 
                'INFO', 
                'FRQ', 
                'SIGNED_SUMSTAT', 
                'NSTUDY']

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def read_header(path):
    '''Return the header (first whitespace‑separated row) of *path*.'''
    openfunc, _ = get_compression(path)
    return [x.rstrip('\n') for x in openfunc(path).readline().split()]


def get_cname_map(flag, default, ignore):
    '''
    Figure out which column names to use.

    Priority is
    (1) ignore everything in ignore
    (2) use everything in flags that is not in ignore
    (3) use everything in default that is not in ignore or in flags

    The keys of flag are cleaned. The entries of ignore are not cleaned. 
    The keys of default are cleaned. But all equality is modulo clean_header().
    '''
    clean_ignore = [clean_header(x) for x in ignore]
    cname_map = {x: flag[x] for x in flag if x not in clean_ignore}
    cname_map.update(
        {x: default[x] for x in default if x not in clean_ignore + flag.keys()})
    return cname_map


def get_compression(path):
    '''
    Read filename suffixes and figure out whether it is gzipped,bzip2'ed or not compressed
    '''
    if path.endswith('gz'):
        return gzip.open, 'gzip'
    if path.endswith('bz2'):
        return bz2.BZ2file, 'bz2'
    return open, None


def clean_header(header):
    '''
    For cleaning file headers.
    - convert to uppercase
    - replace dashes '-' with underscores '_'
    - replace dots '.' (as in R) with underscores '_'
    - remove newlines ('\n')
    '''
    return header.upper().replace('-', '_').replace('.', '_').replace('\n', '')

# -----------------------------------------------------------------------------
# Column/row filters
# -----------------------------------------------------------------------------

def filter_pvals(P, args, log):
    '''Remove out-of-bounds P-values'''
    ii = (P > 0) & (P <= 1)
    bad_p = (~ii).sum()
    if bad_p > 0:
        msg = 'WARNING: {N} SNPs had P outside of (0,1]. The P column may be mislabeled.'
        log.log(msg.format(N=bad_p))

    return ii


def filter_info(info, args, log):
    '''Remove INFO < args.info_min (default 0.9) and complain about out-of-bounds INFO.'''
    if type(info) is pd.Series:  # one INFO column
        oob = ((info > 2.0) | (info < 0)) & info.notnull()
        keep = info >= args.info_min
    elif type(info) is pd.DataFrame:  # several INFO columns
        oob = (((info > 2.0) & info.notnull()).any(axis=1) | (
            (info < 0) & info.notnull()).any(axis=1))
        keep = (info.sum(axis=1) >= args.info_min * (len(info.columns)))
    else:
        raise ValueError('Expected pd.DataFrame or pd.Series.')

    bad_info = oob.sum()
    if bad_info > 0:
        msg = 'WARNING: {N} SNPs had INFO outside of [0,1.5]. The INFO column may be mislabeled.'
        log.log(msg.format(N=bad_info))

    return keep


def filter_frq(frq, args, log):
    '''
    Filter on MAF. Remove MAF < args.maf_min and out-of-bounds MAF.
    '''
    oob = (frq < 0) | (frq > 1)
    bad_frq = oob.sum()
    if bad_frq > 0:
        msg = 'WARNING: {N} SNPs had FRQ outside of [0,1]. The FRQ column may be mislabeled.'
        log.log(msg.format(N=bad_frq))

    frq = np.minimum(frq, 1 - frq)
    keep = frq > args.maf_min
    return keep & ~oob


def filter_alleles(a):
    '''Remove alleles that do not describe strand-unambiguous SNPs'''
    return a.isin(sumstats.VALID_SNPS)

# -----------------------------------------------------------------------------
# Main chunk parser
# -----------------------------------------------------------------------------

def parse_dat(chunk_iter, converted_colname, merge_alleles, log, args):
    '''Parse and filter a sumstats file chunk-wise'''
    tot_snps, kept_chunks = 0, []
    drops = {'NA': 0, 'P': 0, 'INFO': 0,
             'FRQ': 0, 'A': 0, 'SNP': 0, 'MERGE': 0}

    log.log('Reading sumstats from {F} into memory {N} SNPs at a time.'.format(F=args.sumstats, N=int(args.chunksize)))
    
    for chunk in chunk_iter:
        sys.stdout.write('.')
        tot_snps += len(chunk)
        
        # ------------------------------------------------------------------
        # Drop rows with missing data (except INFO / FRQ)
        # ------------------------------------------------------------------
        old = len(chunk)
        subset = [c for c in chunk.columns if converted_colname[c] not in ("INFO", "FRQ")]
        chunk = chunk.dropna(axis=0, how="any", subset=subset)
        drops['NA'] += old - len(chunk)

        chunk.columns = map(lambda x: converted_colname[x], chunk.columns)
 
        wrong_types = [c for c in chunk.columns if c in numeric_cols and not np.issubdtype(chunk[c].dtype, np.number)]
        # If P is of wrong type, fix it by coercing, P<1e-300 is set to 0 rest is set to NaN
        if len(wrong_types) > 0:
            if 'P' in wrong_types:
                bad_mask = ~pd.to_numeric(chunk['P'], errors='coerce').notnull()
                bad_vals = chunk.loc[bad_mask, 'P'].unique()[:10]
                log.log("[DEBUG] Found {} non-numeric P entries (showing up to 10): '{}'".format(bad_mask.sum(), ', '.join(map(str, bad_vals))))
                
                pvals   = pd.to_numeric(chunk['P'], errors='coerce')
                chunk['P'] = pvals
                wrong_types.remove('P')
                if len(wrong_types) > 0:
                    raise ValueError('Columns {} are expected to be numeric'.format(wrong_types))

        keep = np.ones(len(chunk), dtype=bool)

        # Optional allele merge file
        if args.merge_alleles:
            pre = keep.sum()
            keep = chunk.SNP.isin(merge_alleles.SNP)
            drops['MERGE'] += pre - keep.sum()
            if keep.sum() == 0:
                continue
            chunk = chunk[keep].reset_index(drop=True)
            keep = np.ones(len(chunk), dtype=bool)
        
        # INFO quality filtering
        if 'INFO' in chunk.columns:
            pre = keep.sum()
            keep &= filter_info(chunk['INFO'], args, log)
            new = keep.sum()
            drops['INFO'] += pre - new

        # MODIFIED ON 31-10-2023 by David, used to remove rows with missing/invalid FRQ if >50% present
        if 'FRQ' in chunk.columns and chunk['FRQ'].count() > len(chunk['FRQ'])/2:
            pre = keep.sum()
            keep &= filter_frq(chunk['FRQ'], args, log)
            new = keep.sum()
            drops['FRQ'] += pre - new

        # MODIFIED ON 31-10-2023 by David, keep_maf is superseded if all MAFs are missing
        if 'FRQ' in chunk.columns and args.keep_maf and chunk['FRQ'].count() > 0:
            chunk.drop(
                [x for x in ['INFO'] if x in chunk.columns], inplace=True, axis=1)
        else:
            chunk.drop(
                [x for x in ['INFO', 'FRQ'] if x in chunk.columns], inplace=True, axis=1)

        # p‑value sanity check
        pre = keep.sum()
        keep &= filter_pvals(chunk.P, args, log)
        drops['P'] += pre - keep.sum()

        # Allele sanity check
        if not args.no_alleles:
            pre = keep.sum()
            chunk.A1 = chunk.A1.str.upper()
            chunk.A2 = chunk.A2.str.upper()
            keep &= filter_alleles(chunk.A1 + chunk.A2)
            drops['A'] += pre - keep.sum()
        
        if keep.sum() == 0:
            continue

        kept_chunks.append(chunk[keep].reset_index(drop=True))

    sys.stdout.write(' done\n')
    try:
        chunk = pd.concat(kept_chunks, axis=0).reset_index(drop=True)
    except ValueError:
        log.log("ValueError: None of the chunks contain SNPs after filtering")

    # Summary logging
    log_lines = [
        "Read {N} SNPs from --sumstats file.".format(N=tot_snps),
    ]
    if args.merge_alleles:
        log_lines.append("Removed {N} SNPs not in --merge-alleles.".format(N=drops["MERGE"]))
    log_lines.extend([
        "Removed {N} SNPs with missing values.".format(N=drops["NA"]),
        "Removed {N} SNPs with INFO <= {I}.".format(N=drops["INFO"], I=args.info_min),
        "Removed {N} SNPs with MAF <= {M}.".format(N=drops["FRQ"], M=args.maf_min),
        "Removed {N} SNPs with out-of-bounds p-values.".format(N=drops["P"]),
        "Removed {N} variants that were not SNPs or were strand-ambiguous.".format(N=drops["A"]),
        "{N} SNPs remain.".format(N=len(chunk)),
    ])
    log.log("\n".join(log_lines))
    return chunk

# -----------------------------------------------------------------------------
# N / sample‑size handling
# -----------------------------------------------------------------------------

def process_n(dat, args, log):
    '''Determine sample size from --N* flags or N* columns. Filter out low N SNPs.s'''

     # Combine N_CAS/N_CON if both present → N
    if all(i in dat.columns for i in ['N_CAS', 'N_CON']):
        N = dat.N_CAS + dat.N_CON
        P = dat.N_CAS / N
        dat['N'] = N * P / P[N == N.max()].mean()
        dat.drop(['N_CAS', 'N_CON'], inplace=True, axis=1)
        # NB no filtering on N done here -- that is done in the next code block

    # Continuous N column present
    if 'N' in dat.columns:
        n_min = args.n_min if args.n_min else dat.N.quantile(0.9) / 1.5
        old = len(dat)
        dat = dat[dat.N >= n_min].reset_index(drop=True)
        new = len(dat)
        log.log('Removed {M} SNPs with N < {MIN} ({N} SNPs remain).'.format(
            M=old - new, N=new, MIN=n_min))

    # Old‑style NSTUDY column present
    elif 'NSTUDY' in dat.columns and 'N' not in dat.columns:
        nstudy_min = args.nstudy_min if args.nstudy_min else dat.NSTUDY.max()
        old = len(dat)
        dat = dat[dat.NSTUDY >= nstudy_min].drop(
            ['NSTUDY'], axis=1).reset_index(drop=True)
        new = len(dat)
        log.log('Removed {M} SNPs with NSTUDY < {MIN} ({N} SNPs remain).'.format(
            M=old - new, N=new, MIN=nstudy_min))

    # Finally, if no N information in file, fall back to command‑line flags
    if 'N' not in dat.columns:
        if args.N:
            dat['N'] = args.N
            log.log('Using N = {N}'.format(N=args.N))
        elif args.N_cas and args.N_con:
            dat['N'] = args.N_cas + args.N_con
            if args.daner is None:
                msg = 'Using N_cas = {N1}; N_con = {N2}'
                log.log(msg.format(N1=args.N_cas, N2=args.N_con))
        else:
            raise ValueError('Cannot determine N. This message indicates a bug.\n'
                             'N should have been checked earlier in the program.')

    return dat

# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------

def p_to_z(P):
    '''Convert P-value and N to standardized beta.'''
    return np.sqrt(chi2.isf(P, 1))

 
def check_median(x, expected_median, tolerance, name):
    '''Check that median(x) is within tolerance of expected_median.'''
    med = np.median(x)
    if np.abs(med - expected_median) > tolerance:
        msg = 'WARNING: median value of {F} is {V} (should be close to {M}). This column may be mislabeled.'
        raise ValueError(msg.format(F=name, M=expected_median, V=round(med, 2)))

    return 'Median value of {F} was {C}, which seems sensible.'.format(C=med, F=name)

# -----------------------------------------------------------------------------
# Column‑name flag parsing
# -----------------------------------------------------------------------------

def parse_flag_cnames(log, args):
    '''
    Parse flags that specify how to interpret nonstandard column names.

    flag_cnames is a dict that maps (cleaned) arguments to internal column names
    '''
    cname_options = [
        [args.nstudy, 'NSTUDY'],
        [args.snp, 'SNP'],
        [args.N_col, 'N'],
        [args.N_cas_col, 'N_CAS'],
        [args.N_con_col, 'N_CON'],
        [args.a1, 'A1'],
        [args.a2, 'A2'],
        [args.p, 'P'],
        [args.frq, 'FRQ'],
        [args.info, 'INFO']
    ]

    # only include flags specified by user
    flag_cnames = {clean_header(opt[0]): opt[1] for opt in cname_options if opt[0] is not None}

    # --info-list: comma‑separated list of INFO columns
    if args.info_list:
        try:
            flag_cnames.update(
                {clean_header(x): 'INFO' for x in args.info_list.split(',')})
        except ValueError:
            log.log(
                'The argument to --info-list should be a comma-separated list of column names.')
            raise

    # --signed-sumstats: e.g. "Z,0" or "OR,1"
    null_value = None
    if args.signed_sumstats:
        try:
            cname, null_value = args.signed_sumstats.split(',')
            null_value = float(null_value)
            flag_cnames[clean_header(cname)] = 'SIGNED_SUMSTAT'
        except ValueError:
            log.log(
                'The argument to --signed-sumstats should be column header comma number.')
            raise

    return flag_cnames, null_value

# -----------------------------------------------------------------------------
# Allele merge helper
# -----------------------------------------------------------------------------

def allele_merge(dat, alleles, log):
    '''
    WARNING: dat now contains a bunch of NA's~
    Note: dat now has the same SNPs in the same order as --merge alleles.
    '''
    dat = pd.merge(
        alleles, dat, how='left', on='SNP', sort=False).reset_index(drop=True)
    ii = dat.A1.notnull()
    allele_concat = dat.A1[ii] + dat.A2[ii] + dat.MA[ii]
    match = allele_concat.apply(lambda y: y in sumstats.MATCH_ALLELES)
    jj = pd.Series(np.zeros(len(dat), dtype=bool))
    jj[ii] = match
    old = ii.sum()
    n_mismatch = (~match).sum()
    if n_mismatch < old:
        log.log('Removed {M} SNPs whose alleles did not match --merge-alleles ({N} SNPs remain).'.format(M=n_mismatch,
                                                                                                         N=old - n_mismatch))
    else:
        raise ValueError(
            'All SNPs have alleles that do not match --merge-alleles.')

    dat.loc[~jj.astype('bool'), [i for i in dat.columns if i != 'SNP']] = float('nan')
    dat.drop(['MA'], axis=1, inplace=True)
    return dat

# -----------------------------------------------------------------------------
# Navigating harmonized files, Helper functions by David
# -----------------------------------------------------------------------------

def sumstat_picker(cnames, sumstats, prioritize_hm, log):
    """Pick the "best" signed‑sumstat column.

    * If one column has the most non‑missing values, choose that.
    * If several tie **and** *prioritize_hm* is True, prefer a column whose
      name starts with "HM" (case‑insensitive).
    * Otherwise return the first of the tied columns (original LDSC behaviour).
    """
    global HM_PRIORITY_ACTIVE
    df = pd.read_csv(sumstats, sep="\t", compression="gzip", nrows=100, usecols=cnames)
    counts = df.count()
    max_n = counts.max()

    log.log("Found multiple signed summary statistics: {}".format(cnames))
    log.log("Number of non‑missing values in first 100 rows per candidate column:")
    log.log(counts.to_string())

    if prioritize_hm:
        hm_candidates = [c for c in cnames if c.lower().startswith("hm")]
        # require at least 50 non-missing in first 100 rows
        hm_candidates = [c for c in hm_candidates if counts.get(c, 0) >= 50]
        if hm_candidates:
            picked = hm_candidates[0]
            HM_PRIORITY_ACTIVE = True
            log.log("\tSelected '{}' as signed summary statistic\n (forced by '--prioritize-hm')".format(picked))
            return picked

    tied = counts[counts == max_n].index.tolist()
    picked = tied[0]
    
    log.log("\tSelected '{}' as signed summary statistic\n".format(picked))
    return picked

def column_picker(cnames, type, sumstats, log, flag_headers):
    """
    * If a user-specified column exists, keep it.
    * If HM bias is active and an HM_* column has ≥50 % completeness, choose the
      most complete HM column.
    * Otherwise choose the most complete non-HM column (or overall best if only
      HM columns present).
    
    return cnames without duplicate type values.
    """

    dup_cols = [col for col, col_type in cnames.items() if col_type == type]
    log.log("Found multiple {} columns: {}".format(type, dup_cols))
    
    flagged = [c for c in dup_cols if clean_header(c) in flag_headers]
    if flagged:
        picked = flagged[0]
        dup_cols.remove(picked)
        for col in dup_cols:
            cnames.pop(col, None)
        log.log("\tUser-specified column '{}' preserved for {}".format(picked, type))
        return cnames

    df = pd.read_csv(sumstats, sep="\t", compression="gzip", nrows=100, usecols=dup_cols)
    counts = df.count()

    log.log("Number of non‑missing values in first 100 rows per candidate column:")
    log.log(counts.to_string())

    if HM_PRIORITY_ACTIVE == True:
        hm_cols = [c for c in dup_cols if c.lower().startswith("hm")]
        hm_cols = [c for c in hm_cols if counts.get(c, 0) >= 50]
        picked = counts[hm_cols].idxmax() if hm_cols else counts.idxmax()
    else: 
        non_hm_cols = [c for c in dup_cols if not c.lower().startswith("hm")]
        picked = counts[non_hm_cols].idxmax() if non_hm_cols else counts.idxmax()
        
    log.log("\tSelected '{}' as {} column\n".format(picked, type))
    dup_cols.remove(picked)
    for col in dup_cols:
        cnames.pop(col, None)

    return cnames

# -----------------------------------------------------------------------------
# Argument parsing setup
# -----------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument('--sumstats', default=None, type=str,
                    help="Input filename.")
parser.add_argument('--N', default=None, type=float,
                    help="Sample size If this option is not set, will try to infer the sample "
                    "size from the input file. If the input file contains a sample size "
                    "column, and this flag is set, the argument to this flag has priority.")
parser.add_argument('--N-cas', default=None, type=float,
                    help="Number of cases. If this option is not set, will try to infer the number "
                    "of cases from the input file. If the input file contains a number of cases "
                    "column, and this flag is set, the argument to this flag has priority.")
parser.add_argument('--N-con', default=None, type=float,
                    help="Number of controls. If this option is not set, will try to infer the number "
                    "of controls from the input file. If the input file contains a number of controls "
                    "column, and this flag is set, the argument to this flag has priority.")
parser.add_argument('--out', default=None, type=str,
                    help="Output filename prefix.")
parser.add_argument('--info-min', default=0.9, type=float,
                    help="Minimum INFO score.")
parser.add_argument('--maf-min', default=0.01, type=float,
                    help="Minimum MAF.")
parser.add_argument('--daner', default=False, action='store_true',
                    help="Use this flag to parse Stephan Ripke's daner* file format.")
parser.add_argument('--daner-n', default=False, action='store_true',
                    help="Use this flag to parse more recent daner* formatted files, which "
		    "include sample size column 'Nca' and 'Nco'.")
parser.add_argument('--no-alleles', default=False, action="store_true",
                    help="Don't require alleles. Useful if only unsigned summary statistics are available "
                    "and the goal is h2 / partitioned h2 estimation rather than rg estimation.")
parser.add_argument('--merge-alleles', default=None, type=str,
                    help="Same as --merge, except the file should have three columns: SNP, A1, A2, "
                    "and all alleles will be matched to the --merge-alleles file alleles.")
parser.add_argument('--n-min', default=None, type=float,
                    help='Minimum N (sample size). Default is (90th percentile N) / 2.')
parser.add_argument('--chunksize', default=5e6, type=int,
                    help='Chunksize.')

# optional args to specify column names
parser.add_argument('--snp', default=None, type=str,
                    help='Name of SNP column (if not a name that ldsc understands). NB: case insensitive.')
parser.add_argument('--N-col', default=None, type=str,
                    help='Name of N column (if not a name that ldsc understands). NB: case insensitive.')
parser.add_argument('--N-cas-col', default=None, type=str,
                    help='Name of N column (if not a name that ldsc understands). NB: case insensitive.')
parser.add_argument('--N-con-col', default=None, type=str,
                    help='Name of N column (if not a name that ldsc understands). NB: case insensitive.')
parser.add_argument('--a1', default=None, type=str,
                    help='Name of A1 column (if not a name that ldsc understands). NB: case insensitive.')
parser.add_argument('--a2', default=None, type=str,
                    help='Name of A2 column (if not a name that ldsc understands). NB: case insensitive.')
parser.add_argument('--p', default=None, type=str,
                    help='Name of p-value column (if not a name that ldsc understands). NB: case insensitive.')
parser.add_argument('--frq', default=None, type=str,
                    help='Name of FRQ or MAF column (if not a name that ldsc understands). NB: case insensitive.')
parser.add_argument('--signed-sumstats', default=None, type=str,
                    help='Name of signed sumstat column, comma null value (e.g., Z,0 or OR,1). NB: case insensitive.')
parser.add_argument('--info', default=None, type=str,
                    help='Name of INFO column (if not a name that ldsc understands). NB: case insensitive.')
parser.add_argument('--info-list', default=None, type=str,
                    help='Comma-separated list of INFO columns. Will filter on the mean. NB: case insensitive.')
parser.add_argument('--nstudy', default=None, type=str,
                    help='Name of NSTUDY column (if not a name that ldsc understands). NB: case insensitive.')
parser.add_argument('--nstudy-min', default=None, type=float,
                    help='Minimum # of studies. Default is to remove everything below the max, unless there is an N column,'
                    ' in which case do nothing.')
parser.add_argument('--ignore', default=None, type=str,
                    help='Comma-separated list of column names to ignore.')
parser.add_argument('--a1-inc', default=False, action='store_true',
                    help='A1 is the increasing allele.')
parser.add_argument('--keep-maf', default=False, action='store_true',
                    help='Keep the MAF column (if one exists).')
#WIP
parser.add_argument('--prioritize-hm', default=False, action='store_true',
                    help='Prioritize columns with hm_ prefix (GWAS catalog harmonised).')

# -----------------------------------------------------------------------------
# Main function
# -----------------------------------------------------------------------------
# set p = False for testing in order to prevent printing
def munge_sumstats(args, p=True):
    START_TIME = time.time()
    log = Logger(args.out + '.log')

    try:
        if args.out is None:
            raise ValueError('The --out flag is required.')
        if args.sumstats is None:
            raise ValueError('The --sumstats flag is required.')
        if args.no_alleles and args.merge_alleles:
            raise ValueError(
                '--no-alleles and --merge-alleles are not compatible.')
        if args.daner and args.daner_n:
            raise ValueError('--daner and --daner-n are not compatible. Use --daner for sample ' + 
	        'size from FRQ_A/FRQ_U headers, use --daner-n for values from Nca/Nco columns')

        # ------------------------------------------------------------------
        # CLI log header
        # ------------------------------------------------------------------
        if p:
            defaults = vars(parser.parse_args([]))
            opts = vars(args)
            non_defaults = [k for k in opts if opts[k] != defaults[k]]
            header_lines = [MASTHEAD.strip(), "Call: \n", './munge_sumstats.py \\']
            header_lines.extend(['--'+x.replace('_','-')+' '+str(opts[x])+' \\' for x in non_defaults])
            header = "\n".join(header_lines).replace('True','').replace('False','')[0:-1]+'\n'
            log.log(header)

        # ------------------------------------------------------------------
        # Determine column name mapping
        # ------------------------------------------------------------------
        file_cnames = read_header(args.sumstats)  # note keys not cleaned
        flag_cnames, signed_sumstat_null = parse_flag_cnames(log, args)
        if args.ignore:
            ignore_cnames = [clean_header(x) for x in args.ignore.split(',')]
        else:
            ignore_cnames = []

        # If A1 is increasing allele or signed‑sumstats specified, drop implicit defaults
        if args.signed_sumstats is not None or args.a1_inc:
            mod_default_cnames = {k: v for k, v in default_cnames.items() if v not in null_values}
        else:
            mod_default_cnames = default_cnames

        cname_map = get_cname_map(flag_cnames, mod_default_cnames, ignore_cnames)

        # Special legacy daner formats
        if args.daner:
            frq_u = filter(lambda x: x.startswith('FRQ_U_'), file_cnames)[0]
            frq_a = filter(lambda x: x.startswith('FRQ_A_'), file_cnames)[0]
            N_cas = float(frq_a[6:])
            N_con = float(frq_u[6:])
            log.log(
                'Inferred that N_cas = {N1}, N_con = {N2} from the FRQ_[A/U] columns.'.format(N1=N_cas, N2=N_con))
            args.N_cas = N_cas
            args.N_con = N_con
            # drop any N, N_cas, N_con or FRQ columns
            for c in ['N', 'N_CAS', 'N_CON', 'FRQ']:
                for d in [x for x in cname_map if cname_map[x] == 'c']:
                    del cname_map[d]

            cname_map[frq_u] = 'FRQ'

        elif args.daner_n:
            frq_u = filter(lambda x: x.startswith('FRQ_U_'), file_cnames)[0]
            cname_map[frq_u] = 'FRQ'
            try:
                dan_cas = clean_header(file_cnames[file_cnames.index('Nca')])
                dan_con = clean_header(file_cnames[file_cnames.index('Nco')])
            except ValueError:
                raise ValueError('Could not find Nco column expected for daner-n format')
            cname_map[dan_cas] = 'N_CAS'
            cname_map[dan_con] = 'N_CON'

        # Final mapping (original‑header ➜ internal‑name)
        cname_translation = {
            x: cname_map[clean_header(x)] for x in file_cnames if clean_header(x) in cname_map}
        cname_description = {
            x: describe_cname[cname_translation[x]] for x in cname_translation}
        
        # Auto‑detect signed summary statistic column if needed # LEFT OFF MODDING HERE, ONLY REFACTORED AFTER THIS
        if args.signed_sumstats is None and not args.a1_inc:
            sign_cnames = [
                x for x in cname_translation if cname_translation[x] in null_values]
            if len(sign_cnames) > 1:
                # ADDED ON 31-10-2023 by David, used to throw an exception
                picked_cname = sumstat_picker(sign_cnames, args.sumstats, args.prioritize_hm ,log)
                cname_translation = {k: v for k, v in cname_translation.items() if k == picked_cname or k not in sign_cnames}
                sign_cnames = [picked_cname]
            if len(sign_cnames) == 0:
                raise ValueError(
                    'Could not find a signed summary statistic column.')
            sign_cname = sign_cnames[0]
            signed_sumstat_null = null_values[cname_translation[sign_cname]]
            cname_translation[sign_cname] = 'SIGNED_SUMSTAT'
        else:
            sign_cname = 'SIGNED_SUMSTATS'

        # Sanity checks on required columns
        req_cols = ["SNP", "P"] if args.a1_inc else ["SNP", "P", "SIGNED_SUMSTAT"]
        for col in req_cols:
            if col not in cname_translation.values():
                raise ValueError('Could not find {C} column.'.format(C=col))

        # check aren't any duplicated column names in mapping
        for field in cname_translation:
            numk = file_cnames.count(field)
            if numk > 1:
                raise ValueError('Found {num} columns named {C}'.format(C=field,num=str(numk)))

        # check multiple different column names don't map to same data field # PICKER WIP
        for head in cname_translation.values():
            numc = cname_translation.values().count(head)
            if numc > 1:
                cname_translation = column_picker(cname_translation, head, args.sumstats, log, flag_headers = set(flag_cnames.keys()))
            
        # check N handling
        if (not args.N) and (not (args.N_cas and args.N_con)) and ('N' not in cname_translation.values()) and\
                (any(x not in cname_translation.values() for x in ['N_CAS', 'N_CON'])):
            raise ValueError('Could not determine N.')
        if ('N' in cname_translation.values() or all(x in cname_translation.values() for x in ['N_CAS', 'N_CON']))\
                and 'NSTUDY' in cname_translation.values():
            nstudy = [
                x for x in cname_translation if cname_translation[x] == 'NSTUDY']
            for x in nstudy:
                del cname_translation[x]
    
        # check allele columns
        if not args.no_alleles and not all(x in cname_translation.values() for x in ['A1', 'A2']):
            raise ValueError('Could not find A1/A2 columns.')

        # Log column mapping
        log.log('Interpreting column names as follows:')
        log.log('\n'.join([x + ':\t' + cname_description[x]
                           for x in cname_description]) + '\n')

        # Optional allele file for merge
        if args.merge_alleles:
            log.log(
                'Reading list of SNPs for allele merge from {F}'.format(F=args.merge_alleles))
            (openfunc, compression) = get_compression(args.merge_alleles)
            merge_alleles = pd.read_csv(args.merge_alleles, compression=compression, header=0,
                                        delim_whitespace=True, na_values='.')
            if any(x not in merge_alleles.columns for x in ["SNP", "A1", "A2"]):
                raise ValueError(
                    '--merge-alleles must have columns SNP, A1, A2.')

            log.log(
                'Read {N} SNPs for allele merge.'.format(N=len(merge_alleles)))
            merge_alleles['MA'] = (
                merge_alleles.A1 + merge_alleles.A2).apply(lambda y: y.upper())
            merge_alleles = merge_alleles[["SNP", "MA"]]
        else:
            merge_alleles = None

        ###############################################
        # Create iterator over chunks of the input file
        ###############################################
        (openfunc, compression) = get_compression(args.sumstats)

        # figure out which columns are going to involve sign information, so we can ensure
        # they're read as floats
        signed_sumstat_cols = [k for k,v in cname_translation.items() if v=='SIGNED_SUMSTAT']
        dat_gen = pd.read_csv(args.sumstats, delim_whitespace=True, header=0,
                compression=compression, usecols=cname_translation.keys(),
                na_values=['.', 'NA'], iterator=True, chunksize=args.chunksize,
                dtype={c:np.float64 for c in signed_sumstat_cols})

        dat = parse_dat(dat_gen, cname_translation, merge_alleles, log, args)
        if len(dat) == 0:
            raise ValueError('After applying filters, no SNPs remain.')

        # Remove duplicate rs IDs and process N column
        old = len(dat)
        dat = dat.drop_duplicates(subset='SNP').reset_index(drop=True)
        new = len(dat)
        log.log('Removed {M} SNPs with duplicated rs numbers ({N} SNPs remain).'.format(
            M=old - new, N=new))
        dat = process_n(dat, args, log)
        
        # Convert P → Z and flip sign if needed
        dat.P = p_to_z(dat.P)
        dat.rename(columns={'P': 'Z'}, inplace=True)

        if not args.a1_inc:
            log.log(
                check_median(dat.SIGNED_SUMSTAT, signed_sumstat_null, 0.1, sign_cname))
            dat.Z *= (-1) ** (dat.SIGNED_SUMSTAT < signed_sumstat_null)
            dat.drop('SIGNED_SUMSTAT', inplace=True, axis=1)

        # merge last so we don't have to worry about NA values in the rest of
        # the program
        if args.merge_alleles:
            dat = allele_merge(dat, merge_alleles, log)

        # Output
        out_fname = args.out + '.sumstats'
        print_colnames = [
            c for c in dat.columns if c in ['SNP', 'CHR', 'BP', 'N', 'Z', 'A1', 'A2']]
        if args.keep_maf and 'FRQ' in dat.columns:
            print_colnames.append('FRQ')
        msg = 'Writing summary statistics for {M} SNPs ({N} with nonmissing beta) to {F}.'
        log.log(
            msg.format(M=len(dat), F=out_fname + '.gz', N=dat.N.notnull().sum()))
        if p:
            dat.to_csv(out_fname + '.gz', sep="\t", index=False,
                       columns=print_colnames, float_format='%.3f', compression = 'gzip')

        # QC metrics output
        log.log('\nMetadata:')
        CHISQ = dat.Z ** 2
        mean_chisq = CHISQ.mean()
        log.log('Mean chi^2 = ' + str(round(mean_chisq, 3)))
        if mean_chisq < 1.02:
            log.log("WARNING: mean chi^2 may be too small.")

        log.log('Lambda GC = ' + str(round(CHISQ.median() / 0.4549, 3)))
        log.log('Max chi^2 = ' + str(round(CHISQ.max(), 3)))
        log.log('{N} Genome-wide significant SNPs (some may have been removed by filtering).'.format(N=(CHISQ
                                                                                                        > 29).sum()))
        return dat

    except Exception:
        log.log('\nERROR converting summary statistics:\n')
        ex_type, ex, tb = sys.exc_info()
        log.log(traceback.format_exc(ex))
        raise
    finally:
        log.log('\nConversion finished at {T}'.format(T=time.ctime()))
        log.log('Total time elapsed: {T}'.format(
            T=sec_to_str(round(time.time() - START_TIME, 2))))

if __name__ == '__main__':
    munge_sumstats(parser.parse_args(), p=True)
