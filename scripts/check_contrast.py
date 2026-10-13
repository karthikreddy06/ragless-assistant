def lum(hex_code):
    h = hex_code.lstrip('#')
    rgb = [int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4)]
    rgb = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
    return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]

def cr(c1, c2):
    l1 = lum(c1)
    l2 = lum(c2)
    if l1 < l2:
        l1, l2 = l2, l1
    return (l1 + 0.05) / (l2 + 0.05)

pairs = [
    ("Body Text (#3B1F2B) on Surface (#FFFFFF)", "#3B1F2B", "#FFFFFF", "Text >= 4.5:1"),
    ("Body Text (#3B1F2B) on Page Background (#FFF5F8)", "#3B1F2B", "#FFF5F8", "Text >= 4.5:1"),
    ("Body Text (#3B1F2B) on Subtle Pink (#FCE7F0)", "#3B1F2B", "#FCE7F0", "Text >= 4.5:1"),
    ("Muted Text (#7A5566) on Surface (#FFFFFF)", "#7A5566", "#FFFFFF", "Text >= 4.5:1"),
    ("Muted Text (#7A5566) on Page Background (#FFF5F8)", "#7A5566", "#FFF5F8", "Text >= 4.5:1"),
    ("Muted Text (#7A5566) on Subtle Pink (#FCE7F0)", "#7A5566", "#FCE7F0", "Text >= 4.5:1"),
    ("Button Text (#FFFFFF) on Accent (#BE185D)", "#FFFFFF", "#BE185D", "Text >= 4.5:1"),
    ("Accent Link (#BE185D) on Surface (#FFFFFF)", "#BE185D", "#FFFFFF", "Text >= 4.5:1"),
    ("Success Text (#15803D) on Surface (#FFFFFF)", "#15803D", "#FFFFFF", "Text >= 4.5:1"),
    ("Success Text (#15803D) on Success Background (#ECFDF3)", "#15803D", "#ECFDF3", "Text >= 4.5:1"),
    ("Warning Text (#B45309) on Surface (#FFFFFF)", "#B45309", "#FFFFFF", "Text >= 4.5:1"),
    ("Warning Text (#B45309) on Warning Background (#FFF7E6)", "#B45309", "#FFF7E6", "Text >= 4.5:1"),
    ("Error Text (#B91C1C) on Surface (#FFFFFF)", "#B91C1C", "#FFFFFF", "Text >= 4.5:1"),
    ("Error Text (#B91C1C) on Error Background (#FEF2F2)", "#B91C1C", "#FEF2F2", "Text >= 4.5:1"),
    ("Accent Focus Ring (#BE185D) on Surface (#FFFFFF)", "#BE185D", "#FFFFFF", "UI Border >= 3.0:1"),
    ("Input Control Border (#9E5677) on Surface (#FFFFFF)", "#9E5677", "#FFFFFF", "UI Border >= 3.0:1"),
    ("Input Control Border (#9E5677) on Subtle Pink (#FCE7F0)", "#9E5677", "#FCE7F0", "UI Border >= 3.0:1"),
]

print(f"{'Item':<64} | {'Ratio':<8} | {'Standard':<18} | {'Status'}")
print("-" * 102)
all_pass = True
for label, c1, c2, std in pairs:
    ratio = cr(c1, c2)
    min_req = 4.5 if "4.5" in std else 3.0
    status = "PASS" if ratio >= min_req else "FAIL"
    if status == "FAIL":
        all_pass = False
    print(f"{label:<64} | {ratio:.2f}:1  | {std:<18} | {status}")

if not all_pass:
    exit(1)
