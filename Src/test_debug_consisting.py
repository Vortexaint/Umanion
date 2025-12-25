from pathlib import Path
import argparse
import random
import json

from get_stats_region import capture_and_read
from screen_capture import EnergyReader


def generate_debug(year: str = None, month: str = None, turn: int = 1, mood: str = "Neutral", monitor: int = 1, use_capture: bool = True):
    year = year or random.choice(["Junior", "Classic", "Senior"])
    month = month or random.choice(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])

    if use_capture:
        stats = capture_and_read(307, 722, 745, 745, monitor=monitor)
        er = EnergyReader(monitor_num=monitor)
        energy_info = er.read()
        energy = energy_info.get("energy", -1)
    else:
        stats = {"speed": 241, "stamina": 126, "power": 218, "guts": 108, "wit": 106}
        energy = random.randint(0, 100)

    spd = stats.get("speed", -1)
    sta = stats.get("stamina", -1)
    pwr = stats.get("power", -1)
    guts = stats.get("guts", -1)
    wit = stats.get("wit", -1)

    out = {
        "Year": year,
        "Month": month,
        "Turn": turn,
        "Mood": mood,
        "Energy": energy,
        "Stat": {"spd": spd, "sta": sta, "pwr": pwr, "guts": guts, "wit": wit},
    }

    # Print in the requested debug format
    print(f"Year: {year}")
    print(f"Month: {month}")
    print(f"Turn: {turn}")
    print(f"Mood: {mood}")
    print(f"Energy: {energy}")
    print("Stat: {spd} | {sta} | {pwr} | {guts} | {wit}".format(spd=spd, sta=sta, pwr=pwr, guts=guts, wit=wit))

    Path("debug").mkdir(exist_ok=True)
    with open(Path("debug") / "debug_stats.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    return out


def main():
    parser = argparse.ArgumentParser(description="Generate and show a debug " "consisting" " output for stats")
    parser.add_argument("--no-capture", action="store_true", help="Don't capture screen; use sample values")
    parser.add_argument("--turn", type=int, default=1)
    parser.add_argument("--mood", type=str, default="Neutral")
    parser.add_argument("--year", type=str, default=None)
    parser.add_argument("--month", type=str, default=None)
    parser.add_argument("--monitor", type=int, default=1)
    args = parser.parse_args()

    generate_debug(year=args.year, month=args.month, turn=args.turn, mood=args.mood, monitor=args.monitor, use_capture=not args.no_capture)


if __name__ == "__main__":
    main()
