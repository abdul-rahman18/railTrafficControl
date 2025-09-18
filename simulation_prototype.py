import simpy
import random, math, os, csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

class Train:
    def __init__(self, tid, scheduled_entry, speed_mps=16.67, delay_factor=0.0):
        self.id = tid
        self.scheduled_entry = scheduled_entry
        self.speed = speed_mps
        self.delay_factor = delay_factor
        self.clear_event = None
        self.actual_entry = None
        self.exit_time = None
        self.per_block_times = []

def scenario_generator(num_trains=10, start_time=0, headway_sched=300, speed=16.67):
    trains = []
    for i in range(num_trains):
        scheduled = start_time + i * headway_sched
        delay_factor = max(0.0, random.gauss(0.0, 0.25))
        trains.append(Train(tid=(i+1), scheduled_entry=scheduled, speed_mps=speed, delay_factor=delay_factor))
    return trains

def run_simulation(trains, mode='baseline', num_blocks=3, block_lengths=None, sim_time=7200, min_headway=120):
    if block_lengths is None:
        block_lengths = [1000] * num_blocks
    env = simpy.Environment()
    blocks = [simpy.Resource(env, capacity=1) for _ in range(num_blocks)]

    for tr in trains:
        tr.clear_event = env.event()

    events = []

    def train_proc(env, train):
        yield train.clear_event
        train.actual_entry = env.now
        cumulative = 0.0
        for b_idx, blen in enumerate(block_lengths):
            with blocks[b_idx].request() as req:
                yield req
                enter_time = env.now
                travel_time = blen / train.speed
                travel_time *= (1.0 + train.delay_factor)
                yield env.timeout(travel_time)
                exit_time = env.now
                train.per_block_times.append((enter_time, exit_time))
                cumulative += blen
        train.exit_time = env.now
        events.append({
            'id': train.id,
            'scheduled_entry': train.scheduled_entry,
            'actual_entry': train.actual_entry,
            'exit_time': train.exit_time,
            'delay_sec': train.actual_entry - train.scheduled_entry,
            'per_block_times': train.per_block_times
        })

    def baseline_controller(env, trains):
        for tr in trains:
            wait = max(0.0, tr.scheduled_entry - env.now)
            if wait > 0.0:
                yield env.timeout(wait)
            tr.clear_event.succeed()

    def heuristic_controller(env, trains, min_headway):
        last_release = -1e9
        for tr in trains:
            release_time = max(tr.scheduled_entry, last_release + min_headway)
            wait = max(0.0, release_time - env.now)
            if wait > 0.0:
                yield env.timeout(wait)
            tr.clear_event.succeed()
            last_release = release_time

    for tr in trains:
        env.process(train_proc(env, tr))

    if mode == 'baseline':
        env.process(baseline_controller(env, trains))
    else:
        env.process(heuristic_controller(env, trains, min_headway))

    env.run(until=sim_time)

    exited = [e for e in events if e['exit_time'] is not None]
    throughput = len(exited) / (sim_time / 3600.0)  # trains per hour
    avg_delay = np.mean([e['delay_sec'] for e in exited]) if exited else None
    punctuality_pct = (sum(1 for e in exited if e['delay_sec'] <= 300) / len(exited) * 100) if exited else 0.0

    kpis = {'throughput_trains_per_hr': throughput,
            'avg_delay_sec': avg_delay,
            'punctuality_pct': punctuality_pct,
            'trains_exited': len(exited)}
    return events, kpis

def save_events_csv(events, filename):
    rows = []
    for e in events:
        pbt = ";".join([f"{s:.1f}-{x:.1f}" for s,x in e['per_block_times']])
        rows.append({
            'id': e['id'],
            'scheduled_entry': e['scheduled_entry'],
            'actual_entry': e['actual_entry'],
            'exit_time': e['exit_time'],
            'delay_sec': e['delay_sec'],
            'per_block_times': pbt
        })
    df = pd.DataFrame(rows).sort_values('id')
    df.to_csv(filename, index=False)
    print(f"Saved events CSV -> {filename}")

def plot_delay_histogram(events, title, filename):
    delays = [e['delay_sec']/60.0 for e in events]  # minutes
    plt.figure(figsize=(6,4))
    plt.hist(delays, bins=8)
    plt.xlabel('Delay (minutes)')
    plt.ylabel('Number of trains')
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved delay histogram -> {filename}")

def plot_time_space(events, block_lengths, title, filename):
    plt.figure(figsize=(8,6))
    total_len = sum(block_lengths)
    for e in events:
        if not e['per_block_times']:
            continue
        times = []
        dists = []
        cum = 0.0
        times.append(e['actual_entry'])
        dists.append(0.0)
        cum = 0.0
        for (enter, exit) in e['per_block_times']:
            cum += block_lengths[0] if len(block_lengths)>0 else 0 
        cum = 0.0
        for b_idx, (enter, exit) in enumerate(e['per_block_times']):
            times.append(enter)
            dists.append(cum)
            cum += block_lengths[b_idx]
            times.append(exit)
            dists.append(cum)
        plt.plot(np.array(times)/60.0, dists, marker='o', linewidth=1)
        plt.text(times[0]/60.0, 0.02*total_len, f"T{e['id']}", fontsize=8)
    plt.xlabel('Time (minutes)')
    plt.ylabel('Distance along section (m)')
    plt.title(title)
    plt.ylim(-0.05*total_len, total_len*1.05)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved time-space diagram -> {filename}")

def main():
    out_dir = "sih_output"
    os.makedirs(out_dir, exist_ok=True)

    num_trains = 12
    start_time = 0
    scheduled_headway = 180 
    nominal_speed = 20.0   
    trains_base = scenario_generator(num_trains=num_trains, start_time=start_time,
                                     headway_sched=scheduled_headway, speed=nominal_speed)
    import copy
    trains_h = copy.deepcopy(trains_base)
    trains_b = copy.deepcopy(trains_base)

    block_lengths = [800, 800, 800]  

    sim_time = 7200  
    min_headway = 150  

    print("Running BASELINE simulation...")
    events_b, kpis_b = run_simulation(trains_b, mode='baseline', num_blocks=len(block_lengths),
                                      block_lengths=block_lengths, sim_time=sim_time, min_headway=min_headway)
    save_events_csv(events_b, os.path.join(out_dir, "baseline_events.csv"))
    print("Baseline KPIs:", kpis_b)
    plot_delay_histogram(events_b, "Baseline Delay Histogram", os.path.join(out_dir, "baseline_delay_hist.png"))
    plot_time_space(events_b, block_lengths, "Baseline Time-Space", os.path.join(out_dir, "baseline_timespace.png"))

    print("\nRunning HEURISTIC controller simulation...")
    events_h, kpis_h = run_simulation(trains_h, mode='heuristic', num_blocks=len(block_lengths),
                                      block_lengths=block_lengths, sim_time=sim_time, min_headway=min_headway)
    save_events_csv(events_h, os.path.join(out_dir, "heuristic_events.csv"))
    print("Heuristic KPIs:", kpis_h)
    plot_delay_histogram(events_h, "Heuristic Delay Histogram", os.path.join(out_dir, "heuristic_delay_hist.png"))
    plot_time_space(events_h, block_lengths, "Heuristic Time-Space", os.path.join(out_dir, "heuristic_timespace.png"))

    with open(os.path.join(out_dir, "kpis.txt"), "w") as f:
        f.write("BASELINE KPIS:\n")
        f.write(str(kpis_b) + "\n\n")
        f.write("HEURISTIC KPIS:\n")
        f.write(str(kpis_h) + "\n")
    print("\nAll outputs saved in:", out_dir)
    print("Recommendation: open the time-space PNGs and the CSVs for the demo. Use screen-record to show the plots.")

if __name__ == "__main__":
    main()
