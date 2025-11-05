#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Palindrom Midi Song Part Generator
Source: github.com/zeittresor
Requires: pip install mido
"""

import os, sys, random, datetime, math
from typing import List, Tuple, Optional

# ------------- GUI -------------
try:
    import tkinter as tk
    from tkinter import ttk, messagebox
except Exception as e:
    raise RuntimeError("Tkinter is required to run this GUI.") from e

# ------------- MIDI -------------
try:
    from mido import Message, MidiFile, MidiTrack, MetaMessage, bpm2tempo
except ImportError:
    def _err():
        root = tk.Tk(); root.withdraw()
        messagebox.showerror("Missing dependency",
            "The 'mido' package is not installed.\n\nInstall with:\n\n    pip install mido")
        sys.exit(1)
    _err()

# ------------- Theory helpers -------------

MAJOR_STEPS = [0,2,4,5,7,9,11]
HARM_MINOR_STEPS = [0,2,3,5,7,8,11]

GM = {
    "piano":0, "harp":46, "strings":48,
    "violin":40, "cello":42, "oboe":68, "flute":73
}

KEYS = [("C",0,"major"), ("G",7,"major"), ("D",2,"major"), ("F",5,"major"),
        ("A",9,"minor"), ("D",2,"minor"), ("E",4,"minor")]

def pick_key():
    return random.choice(KEYS)

def steps(mode):
    return MAJOR_STEPS if mode=="major" else HARM_MINOR_STEPS

def deg_to_pc(deg:int, tonic_pc:int, mode:str)->int:
    return (tonic_pc + steps(mode)[(deg-1)%7]) % 12

def triad_pcs(deg, tonic_pc, mode):
    s=steps(mode); i=(deg-1)%7
    return [(tonic_pc+s[i])%12,(tonic_pc+s[(i+2)%7])%12,(tonic_pc+s[(i+4)%7])%12]

def nearest_pitch(pc:int, prev:Optional[int], low:int, high:int, prefer:Optional[int]=None)->int:
    cands=[p for p in range(low,high+1) if p%12==pc]
    if not cands:
        return max(low, min(high, prefer if prefer is not None else (low+high)//2))
    if prev is None:
        center = prefer if prefer is not None else (low+high)//2
        return min(cands, key=lambda x:abs(x-center))
    p=min(cands, key=lambda x:abs(x-prev))
    # reduce large leaps by octave shift if possible
    if abs(p-prev)>7:
        for sh in (-12,12,-24,24):
            p2=p+sh
            if low<=p2<=high and abs(p2-prev)<=abs(p-prev):
                return p2
    return p

# ------------- Form & harmony -------------

def theme_progression(mode)->List[int]:
    # 8 bars (16 half-note slots): antecedent (HC) + consequent (PAC)
    ante=[1,1, 2,5, 1,4, 2,5]    # ends on V
    cons=[1,6, 2,5, 1,4, 5,1]    # ends on I
    return ante+cons  # 16 slots

def bridge_progression(mode)->List[int]:
    # 8 bars leaning to V/relativ, then back (safe PD–D–T loops), end on V→I
    return [1,1, 2,5, 1,4, 2,5, 1,3, 4,5, 1,6, 5,1]

def coda_progression(mode)->List[int]:
    # 2 bars I64–V, 2 bars I (plagal tag optional inside pads)
    return [1,1, 5,5, 1,1, 1,1]

# ------------- Macro-contours for melody -------------

def bar_arch_targets(num_bars:int)->List[float]:
    """
    Returns target 'height' per bar (0..1) shaping an arc (A-B-C-B).
    Used to steer register for the bar's strong beats.
    """
    if num_bars<=0: return []
    # Simple archetypes; randomly pick one
    shapes = []
    # Single arch
    shapes.append([0.35 + 0.5*math.sin(math.pi*(i/(num_bars-1))) for i in range(num_bars)])
    # Two small arches
    shapes.append([0.35 + 0.45*math.sin(2*math.pi*(i/(num_bars-1)))**2 for i in range(num_bars)])
    # Rising then plateau then fall
    plateau=max(2, num_bars//3)
    arr=[]
    for i in range(num_bars):
        if i<plateau: arr.append(0.3 + 0.5*(i/max(1,plateau)))
        elif i>num_bars-plateau: arr.append(0.8 - 0.5*((i-(num_bars-plateau))/max(1,plateau)))
        else: arr.append(0.8)
    shapes.append(arr)
    return random.choice(shapes)

# ------------- Rhythm palettes (low density!) -------------

RHYME_HALF_SLOT = [
    [2.0],            # sustained
    [1.0, 1.0],       # two quarters
    [1.5, 0.5],       # dotted + eighth (light)
]

def slot_pattern()->List[float]:
    return random.choices(RHYME_HALF_SLOT, weights=[5,3,2], k=1)[0]

# ------------- Melody builder (cantabile) -------------

def build_cantabile(degrees:List[int], tonic_pc:int, mode:str,
                    low:int=62, high:int=81, center:int=71,
                    max_notes_per_bar:int=3)->List[Tuple[int,float]]:
    """
    For each half-note slot, first sub-event must be chord tone.
    Use stepwise motion, very rare leaps (±3..5) that are resolved by step in opposite direction.
    Density limiter per bar. Macro contour defines target register per bar.
    """
    events=[]
    num_slots=len(degrees)
    num_bars=num_slots//2
    arch=bar_arch_targets(num_bars)

    prev=None
    bar_note_count=0
    curr_bar=0

    for slot in range(num_slots):
        if slot%2==0:
            # new bar
            curr_bar = slot//2
            bar_note_count=0

        chord = triad_pcs(degrees[slot], tonic_pc, mode)
        pat = slot_pattern()
        for j,dur in enumerate(pat):
            if bar_note_count >= max_notes_per_bar:
                # extend last note (legato) instead of adding new ones
                if events:
                    n, last_d = events[-1]
                    events[-1] = (n, last_d + dur)
                continue

            strong = (j==0)
            # Choose target register for this bar from macro-arch
            target_center = int(low + arch[curr_bar]*(high-low))

            if strong:
                # chord tone on strong sub-beat
                pc = min(chord, key=lambda c: 0 if prev is None else min((c-(prev%12))%12, (prev%12 - c)%12))
            else:
                # 70% stepwise passing/neighbor; 30% chord tone
                if random.random()<0.7 and prev is not None:
                    step = random.choice([-2,-1,1,2,-1,1])
                    pc = ( (prev % 12) + step ) % 12
                else:
                    pc = random.choice(chord)

            pitch = nearest_pitch(pc, prev, low, high, target_center)

            # rare expressive leap, then resolve
            if prev is not None and random.random()<0.08 and abs(pitch-prev)<=4:
                leap = random.choice([3,4,5]) * random.choice([-1,1])
                cand = pitch + leap
                if low<=cand<=high:
                    pitch = cand  # take leap now
                    # resolution: fold into next duration if available
                    pass

            # merge tiny repeats into legato
            if events and events[-1][0]==pitch:
                n, last_d = events[-1]
                events[-1] = (n, last_d + dur)
            else:
                events.append((pitch, dur))
                bar_note_count += 1
            prev=pitch

    # enforce final tonic & long hold
    if events:
        final_tonic_pc = deg_to_pc(1, tonic_pc, mode)
        final_p = nearest_pitch(final_tonic_pc, events[-1][0], low, high, center)
        # merge last bar tail into a calm ending
        tail = 0.0
        while events and tail < 2.0:
            p,d = events.pop()
            tail += d
        events.append((final_p, max(2.0, tail)))
    return events

# ------------- Bass / Pad / Sparse Arpeggio -------------

def build_bass(degrees:list, tonic_pc:int, mode:str)->List[Tuple[int,float]]:
    pcs=[deg_to_pc(d,tonic_pc,mode) for d in degrees]
    # Bass on roots, mostly half notes, occasional 1.5 + 0.5 approach
    pitches=[]
    prev=None
    for pc in pcs:
        p=nearest_pitch(pc, prev, 36, 52, 43)
        pitches.append(p); prev=p
    ev=[]
    for i,p in enumerate(pitches):
        if i%4==1 and i+1<len(pitches) and random.random()<0.35:
            nxt=pitches[i+1]; approach = p + (1 if nxt>p else -1)
            if 36<=approach<=52:
                ev.append((p,1.5)); ev.append((approach,0.5)); continue
        ev.append((p,2.0))
    # lengthen last
    if ev: ev[-1] = (ev[-1][0], ev[-1][1] + 0.5)
    return ev

def build_pad(degrees:list, tonic_pc:int, mode:str)->List[Tuple[List[int],float]]:
    out=[]
    for bar in range(0,len(degrees),2):
        pcs = triad_pcs(degrees[bar],tonic_pc,mode)
        notes = sorted({nearest_pitch(pc,None,55,72,62) for pc in pcs})
        out.append((notes,4.0))
    return out

def build_sparse_arp(degrees:list, tonic_pc:int, mode:str)->List[Tuple[int,float]]:
    """Very light arpeggio: at most 4 eighths per bar."""
    out=[]
    for bar in range(0,len(degrees),2):
        pcs = triad_pcs(degrees[bar],tonic_pc,mode)
        low  = nearest_pitch(pcs[0],None,48,60,52)
        mid  = nearest_pitch(pcs[1],low,55,67,60)
        high = nearest_pitch(pcs[2],mid,64,76,69)
        voiced=[low,mid,high]
        pat = random.choice([[0,2,1,2],[0,1,2,1],[0,2,1,2, 2,1]])[:4]  # cap density
        for idx in pat:
            out.append((voiced[idx],0.5))
    return out

# ------------- Tempo & MIDI helpers -------------

def add_tempo_curve(meta:MidiTrack, base_bpm:int, tpq:int, total_beats:float, slow_last_beats:float=6.0):
    base=bpm2tempo(base_bpm)
    meta.append(MetaMessage('set_tempo', time=0, tempo=base))
    if total_beats<=slow_last_beats: return
    start=int((total_beats-slow_last_beats)*tpq)
    meta.append(MetaMessage('set_tempo', time=start, tempo=bpm2tempo(int(base_bpm*0.88))))
    meta.append(MetaMessage('set_tempo', time=int((slow_last_beats-2.0)*tpq), tempo=bpm2tempo(int(max(36, base_bpm*0.72)))))

def add_line(track:MidiTrack, events:List[Tuple[int,float]], tpq:int,
             velocity:int=80, initial_delay:float=0.0,
             crescendo:bool=False, humanize:bool=True):
    pending=int(initial_delay*tpq)
    total=sum(d for _,d in events) or 1.0
    acc=0.0
    for note,beats in events:
        # slight humanization
        vel = velocity
        if crescendo:
            vel = int(50 + 40*(0.5-0.5*math.cos(math.pi*min(1.0,acc/total))))
        if humanize:
            vel = max(1, min(127, vel + random.randint(-3,3)))
        dur = max(1, int(round(beats*tpq * (0.98 if humanize and beats<=0.5 else 1.0))))
        # tiny start offset jitter (convert to extra pending if positive)
        if humanize and random.random()<0.2:
            jitter = random.randint(-3,3)  # ticks
            if jitter>0:
                pending += jitter
            else:
                # starting earlier is tricky with delta times; we shorten prior note instead:
                dur = max(1, dur + jitter)
        track.append(Message('note_on', note=int(note), velocity=vel, time=pending))
        track.append(Message('note_off', note=int(note), velocity=0, time=dur))
        pending=0; acc+=beats
    track.append(MetaMessage('end_of_track', time=0))

def add_poly(track:MidiTrack, chord_events:List[Tuple[List[int],float]], tpq:int,
             velocity:int=56, initial_delay:float=0.0):
    pending=int(initial_delay*tpq)
    for notes,beats in chord_events:
        dur=max(1,int(round(beats*tpq)))
        first=True
        for n in notes:
            track.append(Message('note_on', note=int(n), velocity=velocity, time=pending if first else 0))
            first=False
        first=True
        for n in notes:
            track.append(Message('note_off', note=int(n), velocity=0, time=dur if first else 0))
            first=False
        pending=0
    track.append(MetaMessage('end_of_track', time=0))

# ------------- Piece builder -------------

def build_piece():
    key_name, tonic_pc, mode = pick_key()
    bpm = random.choice([76,84,92,96,104])  # tendenziell ruhiger

    prog_A = theme_progression(mode)        # 16 slots (8 bars)
    prog_B = bridge_progression(mode)       # 16 slots (8 bars)
    prog_Co = coda_progression(mode)        # 8 slots (4 bars)

    # Melodies
    sopr_A = build_cantabile(prog_A, tonic_pc, mode, max_notes_per_bar=3)
    # Bridge: kleine Modulation Richtung Dominante/relativ, aber mit gleicher Kontur
    tonic_dev = (tonic_pc + (7 if mode=="major" else 3)) % 12
    sopr_B = build_cantabile(prog_B, tonic_dev, mode, max_notes_per_bar=3)
    # Reprise zurück in Originaltonart mit leicht erhöhter Mitte
    sopr_Ap = build_cantabile(prog_A, tonic_pc, mode, max_notes_per_bar=3)
    # Coda: sehr ruhig, lange Töne
    sopr_C = build_cantabile(prog_Co, tonic_pc, mode, max_notes_per_bar=2)

    sopr = sopr_A + sopr_B + sopr_Ap + sopr_C

    # Bass / Pad / Sparse Arp
    bass = []
    bass += build_bass(prog_A, tonic_pc, mode)
    bass += build_bass(prog_B, tonic_dev, mode)
    bass += build_bass(prog_A, tonic_pc, mode)
    bass += build_bass(prog_Co, tonic_pc, mode)

    pad = []
    pad += build_pad(prog_A, tonic_pc, mode)
    pad += build_pad(prog_B, tonic_dev, mode)
    pad += build_pad(prog_A, tonic_pc, mode)
    pad += build_pad(prog_Co, tonic_pc, mode)

    arp = []
    # nur in A und B sporadisch, Coda ohne
    if random.random()<0.6:
        arp += build_sparse_arp(prog_A, tonic_pc, mode)
    if random.random()<0.6:
        arp += build_sparse_arp(prog_B, tonic_dev, mode)

    return {
        "key": key_name, "tonic_pc": tonic_pc, "mode": mode, "bpm": bpm,
        "sopr": sopr, "bass": bass, "pad": pad, "arp": arp
    }

# ------------- Save MIDI -------------

def filename(form, key, mode, bpm):
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{form}_{key}_{mode}_BPM{bpm}_{ts}.mid".replace(" ","")

def save_piece(piece):
    mid = MidiFile()
    tpq = mid.ticks_per_beat

    total_beats = sum(d for _,d in piece["sopr"])
    meta = MidiTrack(); mid.tracks.append(meta)
    add_tempo_curve(meta, piece["bpm"], tpq, total_beats, slow_last_beats=6.0)
    meta.append(MetaMessage('time_signature', numerator=4, denominator=4,
                            clocks_per_click=24, notated_32nd_notes_per_beat=8, time=0))

    # Track 1: Lead (Oboe/Flute early; Violin later with long notes)
    t1 = MidiTrack(); mid.tracks.append(t1)
    lead_prog = random.choice([GM["oboe"], GM["flute"]])
    t1.append(Message('program_change', program=lead_prog, time=0))
    add_line(t1, piece["sopr"], tpq, velocity=86, crescendo=True, humanize=True)

    # Track 2: Strings pad (sustain)
    t2 = MidiTrack(); mid.tracks.append(t2)
    t2.append(Message('program_change', program=GM["strings"], time=0))
    add_poly(t2, piece["pad"], tpq, velocity=52)

    # Track 3: Sparse arpeggio (Piano/Harp chosen by key/mode)
    if piece["arp"]:
        t3 = MidiTrack(); mid.tracks.append(t3)
        t3.append(Message('program_change', program=(GM["harp"] if piece["mode"]=="minor" else GM["piano"]), time=0))
        add_line(t3, piece["arp"], tpq, velocity=56, crescendo=False, humanize=True)

    # Track 4: Bass (Cello)
    t4 = MidiTrack(); mid.tracks.append(t4)
    t4.append(Message('program_change', program=GM["cello"], time=0))
    add_line(t4, piece["bass"], tpq, velocity=70, crescendo=False, humanize=False)

    name = filename("NarrativePiece", piece["key"], piece["mode"], piece["bpm"])
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), name)
    mid.save(path)
    return path

# ------------- GUI -------------

def main():
    root = tk.Tk()
    root.title("Narrative Classical MIDI Generator")

    fixed_w, fixed_h = 560, 140
    root.geometry(f"{fixed_w}x{fixed_h}")
    root.minsize(fixed_w, fixed_h)
    root.resizable(False, False)

    frm = ttk.Frame(root, padding=12); frm.grid(row=0,column=0,sticky="nsew")
    root.columnconfigure(0,weight=1); root.rowconfigure(0,weight=1)

    status = tk.StringVar(value="Ready.")
    btn = ttk.Button(frm, text="Generate"); btn.grid(row=0,column=0,sticky="we", padx=2, pady=(0,8))
    lbl = ttk.Label(frm, textvariable=status, anchor="w", wraplength=fixed_w-32, justify="left")
    lbl.grid(row=1,column=0,sticky="we")

    def on_generate():
        try:
            piece = build_piece()
            path  = save_piece(piece)
            status.set(f"Saved: {os.path.basename(path)}  ({piece['key']} {piece['mode']}, {piece['bpm']} BPM)")
        except Exception as e:
            messagebox.showerror("Generation error", f"{type(e).__name__}: {e}")

    btn.configure(command=on_generate)
    root.update_idletasks()
    w,h = root.winfo_width(), root.winfo_height()
    root.geometry(f"{w}x{h}"); root.minsize(w,h)

    root.mainloop()

if __name__=="__main__":
    main()
