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
    ("Light: Body Text (#1C2F26) on Card Surface (#FFFFFF)", "#1C2F26", "#FFFFFF", "Text >= 4.5:1"),
    ("Light: Body Text (#1C2F26) on Page Background (#F8F6F3)", "#1C2F26", "#F8F6F3", "Text >= 4.5:1"),
    ("Light: Body Text (#1C2F26) on Chip Blush (#FBE9F0)", "#1C2F26", "#FBE9F0", "Text >= 4.5:1"),
    ("Light: Muted Text (#415B4E) on Card Surface (#FFFFFF)", "#415B4E", "#FFFFFF", "Text >= 4.5:1"),
    ("Light: Button Text (#FFFFFF) on Primary Button (#1A78C2)", "#FFFFFF", "#1A78C2", "Text >= 4.5:1"),
    ("Light: Primary Text / Links (#115E9B) on Surface (#FFFFFF)", "#115E9B", "#FFFFFF", "Text >= 4.5:1"),
    ("Light: Success Text (#286036) on Card Surface (#FFFFFF)", "#286036", "#FFFFFF", "Text >= 4.5:1"),
    ("Light: Success Text (#286036) on Success Subtle (#EAF4ED)", "#286036", "#EAF4ED", "Text >= 4.5:1"),
    ("Light: Warning Text (#943E15) on Card Surface (#FFFFFF)", "#943E15", "#FFFFFF", "Text >= 4.5:1"),
    ("Light: Warning Text (#943E15) on Warning Subtle (#FDF0EA)", "#943E15", "#FDF0EA", "Text >= 4.5:1"),
    ("Light: Danger Text (#A0272B) on Card Surface (#FFFFFF)", "#A0272B", "#FFFFFF", "Text >= 4.5:1"),
    ("Light: Pink UI Border (#C05D87) on Blush Chip (#FBE9F0)", "#C05D87", "#FBE9F0", "UI Border >= 3.0:1"),
    ("Light: Pink UI Border (#C05D87) on Card Surface (#FFFFFF)", "#C05D87", "#FFFFFF", "UI Border >= 3.0:1"),
    ("Light: Card Border (#DDD7D0) on Page (#F8F6F3)", "#857C74", "#F8F6F3", "UI Border >= 3.0:1"),
    ("Dark: Body Text (#F1F5F3) on Card Surface (#182C24)", "#F1F5F3", "#182C24", "Text >= 4.5:1"),
    ("Dark: Body Text (#F1F5F3) on Page Background (#12201A)", "#F1F5F3", "#12201A", "Text >= 4.5:1"),
    ("Dark: Muted Text (#A2BFB2) on Card Surface (#182C24)", "#A2BFB2", "#182C24", "Text >= 4.5:1"),
    ("Dark: Button Text (#0E1B15) on Primary Button (#58A6E6)", "#0E1B15", "#58A6E6", "Text >= 4.5:1"),
    ("Dark: Primary Link Text (#72B6ED) on Card Surface (#182C24)", "#72B6ED", "#182C24", "Text >= 4.5:1"),
    ("Dark: Success Text (#68C47F) on Card Surface (#182C24)", "#68C47F", "#182C24", "Text >= 4.5:1"),
    ("Dark: Warning Text (#F29C6D) on Card Surface (#182C24)", "#F29C6D", "#182C24", "Text >= 4.5:1"),
    ("Dark: Danger Text (#F58585) on Card Surface (#182C24)", "#F58585", "#182C24", "Text >= 4.5:1"),
    ("Dark: Pink UI Border (#B06588) on Dark Chip (#271B23)", "#B06588", "#271B23", "UI Border >= 3.0:1"),
    ("Dark: Pink UI Border (#B06588) on Card Surface (#182C24)", "#B06588", "#182C24", "UI Border >= 3.0:1"),
    ("Dark: Accent Pink Text (#F3AEC9) on Card Surface (#182C24)", "#F3AEC9", "#182C24", "Text >= 4.5:1"),
]

print(f"{'Item':<62} | {'Ratio':<8} | {'Standard':<18} | {'Status'}")
print("-" * 100)
for label, c1, c2, std in pairs:
    ratio = cr(c1, c2)
    min_req = 4.5 if "4.5" in std else 3.0
    status = "PASS" if ratio >= min_req else "FAIL"
    print(f"{label:<62} | {ratio:.2f}:1  | {std:<18} | {status}")
