#!/usr/bin/env python3
from __future__ import annotations
import csv, json, os, re, signal, sys, time, zipfile, shutil
from collections import defaultdict
from pathlib import Path

AUDIO_EXT={'.wav','.aif','.aiff','.flac','.ogg','.au'}
PROJECT=Path('/mnt/data/work_current/Aaron_Sound_Sorter')
ZIP=Path('/mnt/data/one_shot_percussive_sounds.zip')
ROOT=Path(os.environ.get('PERC_SYSTEM_ROOT', '/mnt/data/perc_system_run'))
EXTRACT=ROOT/'neutral_extract'
RESULTS=ROOT/'results.csv'
MAP=ROOT/'member_map.csv'
STATE=ROOT/'state.json'

sys.path.insert(0,str(PROJECT))
sys.path.insert(0,str(PROJECT/'src'))

class Timeout(Exception): pass
def handler(signum, frame): raise Timeout('sample timeout')
signal.signal(signal.SIGALRM, handler)

def clean_member(m):
    parts=[p for p in m.replace('\\','/').split('/') if p]
    return parts and '__MACOSX' not in parts and not any(p.startswith('._') or p.startswith('.') for p in parts) and Path(parts[-1]).suffix.lower() in AUDIO_EXT

def enumerate_members():
    by=defaultdict(list)
    with zipfile.ZipFile(ZIP) as z:
        for info in z.infolist():
            if info.is_dir() or not clean_member(info.filename): continue
            folder=str(Path(info.filename).parent)
            by[folder].append(info.filename)
    for k in by: by[k].sort()
    return dict(sorted(by.items()))

def selected_members(batch_start:int, per_folder:int):
    by=enumerate_members()
    rows=[]
    for g,(folder,members) in enumerate(by.items(),1):
        for j,m in enumerate(members[batch_start:batch_start+per_folder], batch_start+1):
            rows.append((g,folder,j,m))
    return rows

def extract_rows(rows, force=False):
    EXTRACT.mkdir(parents=True, exist_ok=True)
    rows_out=[]
    with zipfile.ZipFile(ZIP) as z:
        for global_i,(g,folder,j,m) in enumerate(rows,1):
            ext=Path(m).suffix.lower() or '.wav'
            # neutral names; preserve group only, no source instrument words
            dest=EXTRACT/f'g{g:02d}'/f'neutral_g{g:02d}_{j:05d}{ext}'
            dest.parent.mkdir(parents=True, exist_ok=True)
            if force or not dest.exists() or dest.stat().st_size == 0:
                with z.open(m) as src, dest.open('wb') as out:
                    shutil.copyfileobj(src,out)
            rows_out.append({'group':g,'folder':folder,'ordinal':j,'member':m,'neutral_path':str(dest)})
    ROOT.mkdir(parents=True, exist_ok=True)
    with MAP.open('w',newline='') as f:
        w=csv.DictWriter(f, fieldnames=['group','folder','ordinal','member','neutral_path'])
        w.writeheader(); w.writerows(rows_out)
    return rows_out

def load_sorter():
    from Aaron_Sound_Sorter import declare_voters, build_sorter
    from aaron_sound_sorter.infrastructure.brain_repository import BrainRepository
    repo=BrainRepository()
    brain=repo.load(PROJECT/'stage4_folder_brain.json')
    baby={}
    for lane,name in [('core_baby','stage4_folder_brain_core_baby.json'),('spread_baby','stage4_folder_brain_spread_baby.json'),('outlier_baby','stage4_folder_brain_outlier_baby.json')]:
        p=PROJECT/name
        baby[lane]=repo.load(p) if p.exists() else None
    sorter=build_sorter(declare_voters(candidate_count=100), candidate_count=100)
    return sorter, brain, baby

def classify_rows(rows, start_at=0, max_rows=None, timeout=45):
    sorter, brain, baby = load_sorter()
    fields=['seq','group','folder','ordinal','member','neutral_file','path','top','ok_broad','review','instrument_steal','status','final_source','shape','raw_consensus','raw_family','seconds','error']
    existing=[]
    done_neutral=set()
    if RESULTS.exists():
        with RESULTS.open() as f:
            for r in csv.DictReader(f):
                existing.append(r); done_neutral.add(r.get('neutral_file',''))
    outfh=RESULTS.open('a',newline='')
    w=csv.DictWriter(outfh, fieldnames=fields, extrasaction='ignore')
    if not existing: w.writeheader()
    count=0
    for seq,row in enumerate(rows,1):
        if seq <= start_at: continue
        if max_rows is not None and count>=max_rows: break
        neutral=str(row['neutral_path'])
        if neutral in done_neutral: continue
        p=Path(neutral); t=time.time()
        rec={'seq':seq,'group':row['group'],'folder':row['folder'],'ordinal':row['ordinal'],'member':row['member'],'neutral_file':neutral}
        try:
            signal.alarm(timeout)
            r=sorter.classify_one_file(p, brain, baby, {}, use_baby_brains_in_sort=True, use_harmonic_brains_in_sort=False)
            signal.alarm(0)
            d=r.decision
            path=getattr(d,'folder_path','') or getattr(d,'final_label','') or ''
            ev=getattr(r.facts,'evidence',{}) or {}
            auth=ev.get('authority_trace',{}) if isinstance(ev.get('authority_trace',{}),dict) else {}
            shape=ev.get('shape_vote',{}) if isinstance(ev.get('shape_vote',{}),dict) else {}
            raw=ev.get('raw_consensus_claim',{}) if isinstance(ev.get('raw_consensus_claim',{}),dict) else {}
            top=path.split('/')[0] if path else ''
            rec.update({
                'path':path,
                'top':top,
                'ok_broad': str(path.startswith('Drums/') or path.startswith('FX/')),
                'review': str(path.startswith('_TO_REVIEW')),
                'instrument_steal': str(path.startswith('Instruments/')),
                'status':getattr(d,'consensus_status',''),
                'final_source':auth.get('final_source',''),
                'shape':shape.get('shape',''),
                'raw_consensus':raw.get('path','') or raw.get('label','') or '',
                'raw_family':raw.get('family',''),
                'seconds':round(time.time()-t,2),
                'error':''
            })
            print(f"{seq:04d} g{row['group']} {Path(row['member']).name[:55]:55s} -> {path} ({rec['seconds']}s) ok={rec['ok_broad']}", flush=True)
        except Exception as e:
            signal.alarm(0)
            rec.update({'path':'','top':'','ok_broad':'False','review':'False','instrument_steal':'False','seconds':round(time.time()-t,2),'error':repr(e)})
            print(f"ERR {seq:04d} g{row['group']} {Path(row['member']).name[:55]:55s} {repr(e)}", flush=True)
        w.writerow(rec); outfh.flush(); count+=1
    outfh.close()

def summarize():
    if not RESULTS.exists(): return
    rows=list(csv.DictReader(RESULTS.open()))
    bad=[r for r in rows if r.get('ok_broad')!='True']
    review=[r for r in rows if r.get('review')=='True']
    inst=[r for r in rows if r.get('instrument_steal')=='True']
    print('\nSUMMARY')
    print('tested',len(rows),'bad',len(bad),'review',len(review),'instrument',len(inst))
    by=defaultdict(lambda:[0,0,0,0])
    for r in rows:
        a=by[r['folder']]; a[0]+=1; a[1]+= (r.get('ok_broad')=='True'); a[2]+=(r.get('review')=='True'); a[3]+=(r.get('instrument_steal')=='True')
    for folder,a in by.items(): print(a, folder)
    if bad:
        print('\nBAD FIRST 30')
        for r in bad[:30]: print(r['seq'],r['member'],'=>',r.get('path'),r.get('error'), 'shape', r.get('shape'), 'source', r.get('final_source'))

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument('--batch-start',type=int,default=0, help='0-based offset within each source folder')
    ap.add_argument('--per-folder',type=int,default=5)
    ap.add_argument('--max-rows',type=int,default=None)
    ap.add_argument('--reset',action='store_true')
    args=ap.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    if args.reset and RESULTS.exists(): RESULTS.unlink()
    rows=extract_rows(selected_members(args.batch_start,args.per_folder))
    print('selected',len(rows),'from',len(set(r[1] for r in selected_members(args.batch_start,args.per_folder))),'folders')
    classify_rows(rows, max_rows=args.max_rows)
    summarize()
