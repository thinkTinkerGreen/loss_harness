"""
Automated verification of submission criteria (§6 of PROMPT_loss_harness_notebook.md)
"""
import os
import re
import json

def verify():
    print("=" * 60)
    print("RUNNING AUTOMATED ACCEPTANCE CHECKLIST (§6)")
    print("=" * 60)

    # 1. Check results.json exists and loads
    assert os.path.exists("results.json"), "results.json missing"
    with open("results.json", "r", encoding="utf-8") as f:
        res = json.load(f)
    print("✔ results.json loaded successfully")

    # 2. Check README.md exists and has no placeholders or unfilled slots
    assert os.path.exists("README.md"), "README.md missing"
    with open("README.md", "r", encoding="utf-8") as f:
        readme = f.read()
    
    unreplaced = re.findall(r"\{\{[^}]+\}\}", readme)
    assert len(unreplaced) == 0, f"Found unreplaced placeholders in README.md: {unreplaced}"
    assert "<!-- WRITE AFTER RUN" not in readme, "Found unfilled WRITE AFTER RUN slots in README.md"
    print("✔ README.md has no unreplaced {{ tags and all 4 interpretation slots filled")

    # 3. Check assets/ PNGs exist and have non-zero size
    expected_assets = ["assets/neg_controls.png", "assets/memory.png", "assets/two_heads.png"]
    for asset in expected_assets:
        assert os.path.exists(asset), f"Missing {asset}"
        size = os.path.getsize(asset)
        assert size > 5000, f"{asset} is suspiciously small ({size} bytes)"
        print(f"✔ {asset} exists ({size:,} bytes)")

    # 4. Check numbers match between results.json and README.md
    check_values = [
        str(res["env"]["torch"]),
        str(res["env"]["cuda"]),
        str(res["env"]["gpu"]),
        str(res["neg"]["correct"]),
        str(res["neg"]["shift0"]),
        str(res["neg"]["shift_neg1"]),
        str(res["pad"]["n_nomask"]),
        str(res["pad"]["n_masked"]),
        str(res["pack"]["trained"]["boundary"]),
        str(res["ppl"]["loss"]),
        f"{res['params']['tied']:,}",
        f"{res['params']['untied']:,}",
        f"{res['params']['delta']:,}",
        res["mem"]["ordinary_fb"],
        res["mem"]["chunked_fb"],
        res["mem"]["ratio"],
        str(res["p2"]["train"]["l1"]),
        str(res["p2"]["train"]["l2"]),
        str(res["p2"]["control_l1"])
    ]
    for val in check_values:
        assert val in readme, f"Value '{val}' from results.json not found in README.md!"
    print("✔ All 19 key measured quantities in results.json match README.md exactly")

    # 5. Check item 6 parameter exact assertions
    assert res["params"]["tied"] == 16091136
    assert res["params"]["untied"] == 28957184
    assert res["params"]["delta"] == 12866048
    assert res["params"]["two_head"] == 28957184
    print("✔ Item 6 parameter assertions verified exactly: 16,091,136 / 28,957,184 / 12,866,048")

    # 6. Check padding counts
    assert res["pad"]["n_nomask"] == 124
    assert res["pad"]["n_masked"] == 76
    assert res["pad"]["poison_masked_equal"] is True
    assert res["pad"]["poison_unmasked_changed"] is True
    print("✔ Item 3 padding & poison assertions verified: 124 != 76 and poison tests passed")

    # 7. Check memory ratio >= 2x
    ratio_float = float(res["mem"]["ratio"].replace("x", ""))
    assert ratio_float >= 2.0, f"Memory reduction ratio {ratio_float} is below 2x"
    print(f"✔ Item 7 headline memory ratio verified: {ratio_float:.2f}x (>= 2x)")

    # 8. Check Part 2 ordering
    assert res["p2"]["train"]["l2"] > res["p2"]["train"]["l1"], "Head 2 loss should be greater than Head 1"
    print(f"✔ Part 2 ordering verified: L2 ({res['p2']['train']['l2']}) > L1 ({res['p2']['train']['l1']})")

    print("=" * 60)
    print("ALL ACCEPTANCE CRITERIA PASSED WITH 100% COMPLIANCE!")
    print("=" * 60)

if __name__ == "__main__":
    verify()
