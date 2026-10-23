import logging
import re
from datetime import datetime

from data_base import create_data_base, save_event
from bin_rules import BIN_IDS, ITEM_TO_BIN, resolve_bin

logger = logging.getLogger("rewa")

# Single source of truth for Bins and Mappings
def create_bins():
    return {
        name: {
            "bin_id": bin_id,
            "item_count": 0,
            "capacity_percent": 0,
            "times_emptied": 0,
            "history": [],
        }
        for name, bin_id in BIN_IDS.items()
    }


bins = create_bins()


def get_bin_for_item(item_name):
    return resolve_bin(item_name)


# 3. Record a new recycling event
def record_item(waste_type, item_name, state=None, persist=True):
    """Add one item to a bin.

    persist=True  -> also save the event in the database (logged-in users).
    persist=False -> bin only (guests: nothing is saved).

    The database write happens FIRST. If it fails, the bin is left
    untouched, so the bins and the saved history can never disagree.
    Returns True on success, False otherwise.
    """
    if state is None:
        state = bins

    # The category chosen by the user wins; item name is only a fallback
    target = waste_type if waste_type in state else get_bin_for_item(item_name)

    if target not in state:
        logger.warning("No bin for '%s' (got '%s'). Not recorded.",
                       item_name, waste_type)
        return False

    selected_bin = state[target]

    if selected_bin["capacity_percent"] >= 100:
        logger.warning("%s bin is full! Please empty it first.", target)
        return False

    now = datetime.now()

    if persist:
        try:
            create_data_base()
            save_event(
                material=target,
                item_name=item_name,
                timestamp=now.isoformat(timespec="seconds"),
            )
        except Exception as error:
            logger.warning("Could not save recycling event: %s", error)
            return False

    selected_bin["item_count"] += 1

    # Simulated capacity, not a real measurement
    selected_bin["capacity_percent"] = min(
        selected_bin["capacity_percent"] + 5, 100
    )

    selected_bin["history"].append({
        "item": item_name,
        "waste_type": target,
        "time": now.strftime("%H:%M:%S"),
    })
    return True


# 4. Empty a bin
def empty_bin(waste_type, state=None):
    if state is None:
        state = bins
    if waste_type not in state:
        logger.warning("Unknown bin: %s", waste_type)
        return False

    b = state[waste_type]
    b["item_count"] = 0
    b["capacity_percent"] = 0
    b["history"] = []
    b["times_emptied"] += 1
    return True


def empty_all_bins(state=None):
    if state is None:
        state = bins
    for name in state:
        empty_bin(name, state)


# 5. Dashboard (console)
def show_dashboard():
    print("\n===== SMART RECYCLING DIGITAL TWIN =====")
    for waste_type, data in bins.items():
        print(f"\nBin: {waste_type}")
        print(f"ID: {data['bin_id']}")
        print(f"Items: {data['item_count']}")
        print(f"Estimated capacity: {data['capacity_percent']}%")
        print(f"Times emptied: {data['times_emptied']}")


# Run test events only when this file is run directly
if __name__ == "__main__":
    record_item("Plastic", "Plastic bottle")
    record_item("Paper", "Cardboard box")
    record_item("Hazardous", "Battery")
    record_item("Biodegradable", "Fruit peel")
    show_dashboard()
    empty_bin("Paper")
    show_dashboard()