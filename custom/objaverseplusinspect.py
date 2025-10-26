from datasets import load_dataset
import objaverse


ds = load_dataset("cindyxl/ObjaversePlusPlus", split="train")
uids = ds["UID"]               
total = len(uids)

high_quality = ds.filter(lambda x: x["score"] >= 2)
uids = high_quality["UID"]             
print(f"Number of 2 or higher: {len(uids)}")
print("\n")
#very high quality
very_high_quality = ds.filter(lambda x: x["score"] >= 3)
uids = very_high_quality["UID"]               
print(f"Number of 3 or higher: {len(uids)}")
print("\n")
no_multi = very_high_quality.filter(lambda x: x["is_multi_object"] == "false")
uids = no_multi["UID"]             
print(f"Number of 3 or higher and no multi-object: {len(uids)}")
print("\n")
#filter out single color ones aswell
no_single_color = no_multi.filter(lambda x: x["is_single_color"] == "false")
uids = no_single_color["UID"]              
print(f"Number of 3 or higher and no multi-object and no single color: {len(uids)}")
print("\n")

TOTAL = len(uids)
print(f"Total number of objects: {TOTAL}")
