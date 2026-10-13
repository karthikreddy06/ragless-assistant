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
    ("Primary Text (#382936) on Surface (#FFFFFF)", "#382936", "#FFFFFF", "Text >= 4.5:1"),
    ("Primary Text (#382936) on Page Background (#FDF7FA)", "#382936", "#FDF7FA", "Text >= 4.5:1"),
    ("Primary Text (#382936) on Soft Pink (#F4C6D8)", "#382936", "#F4C6D8", "Text >= 4.5:1"),
    ("Primary Text (#382936) on Light Blush (#F9E3EC)", "#382936", "#F9E3EC", "Text >= 4.5:1"),
    ("Primary Text (#382936) on Sky Blue (#B9E5F5)", "#382936", "#B9E5F5", "Text >= 4.5:1"),
    ("Primary Text (#382936) on Pale Blue (#D9F1FA)", "#382936", "#D9F1FA", "Text >= 4.5:1"),
    ("Secondary Text (#796B79) on Surface (#FFFFFF)", "#796B79", "#FFFFFF", "Text >= 4.5:1"),
    ("Secondary Text (#796B79) on Page Background (#FDF7FA)", "#796B79", "#FDF7FA", "Text >= 4.5:1"),
    ("Accent Hover (#AD507F) on Surface (#FFFFFF)", "#AD507F", "#FFFFFF", "Text >= 4.5:1"),
    ("Success Text (#286036) on Surface (#FFFFFF)", "#286036", "#FFFFFF", "Text >= 4.5:1"),
    ("Success Text (#286036) on Success Subtle (#EDF7EE)", "#286036", "#EDF7EE", "Text >= 4.5:1"),
    ("Warning Text (#8A4E12) on Surface (#FFFFFF)", "#8A4E12", "#FFFFFF", "Text >= 4.5:1"),
    ("Warning Text (#8A4E12) on Warning Subtle (#FFF8E1)", "#8A4E12", "#FFF8E1", "Text >= 4.5:1"),
    ("Error Text (#991B1B) on Surface (#FFFFFF)", "#991B1B", "#FFFFFF", "Text >= 4.5:1"),
    ("Error Text (#991B1B) on Error Subtle (#FEE2E2)", "#991B1B", "#FEE2E2", "Text >= 4.5:1"),
    ("Focus Ring / Primary Accent (#C76B98) on Surface (#FFFFFF)", "#C76B98", "#FFFFFF", "UI Border >= 3.0:1"),
    ("Input Control Border (#8E6B83) on Surface (#FFFFFF)", "#8E6B83", "#FFFFFF", "UI Border >= 3.0:1"),
    ("Input Control Border (#8E6B83) on Soft Pink (#F4C6D8)", "#8E6B83", "#F4C6D8", "UI Border >= 3.0:1"),
    ("Input Control Border (#8E6B83) on Light Blush (#F9E3EC)", "#8E6B83", "#F9E3EC", "UI Border >= 3.0:1"),
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
