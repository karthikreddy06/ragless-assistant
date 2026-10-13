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
    ("Light: Body Text (#0F172A) on Surface (#FFFFFF)", "#0F172A", "#FFFFFF", "Text >= 4.5:1"),
    ("Light: Body Text (#0F172A) on Page (#F8FAFC)", "#0F172A", "#F8FAFC", "Text >= 4.5:1"),
    ("Light: Muted Text (#64748B) on Surface (#FFFFFF)", "#64748B", "#FFFFFF", "Text >= 4.5:1"),
    ("Light: Muted Text (#64748B) on Page (#F8FAFC)", "#64748B", "#F8FAFC", "Text >= 4.5:1"),
    ("Light: Button Text (#FFFFFF) on Primary Accent (#2563EB)", "#FFFFFF", "#2563EB", "Text >= 4.5:1"),
    ("Light: Primary Accent Link (#1D4ED8) on Surface (#FFFFFF)", "#1D4ED8", "#FFFFFF", "Text >= 4.5:1"),
    ("Light: Verified/Green Text (#15803D) on Surface (#FFFFFF)", "#15803D", "#FFFFFF", "Text >= 4.5:1"),
    ("Light: Verified/Green Text (#15803D) on Subtle Green (#F0FDF4)", "#15803D", "#F0FDF4", "Text >= 4.5:1"),
    ("Light: Review/Amber Text (#B45309) on Surface (#FFFFFF)", "#B45309", "#FFFFFF", "Text >= 4.5:1"),
    ("Light: Review/Amber Text (#B45309) on Subtle Amber (#FFFBEB)", "#B45309", "#FFFBEB", "Text >= 4.5:1"),
    ("Light: Error/Red Text (#B91C1C) on Surface (#FFFFFF)", "#B91C1C", "#FFFFFF", "Text >= 4.5:1"),
    ("Light: Error/Red Text (#B91C1C) on Subtle Red (#FEF2F2)", "#B91C1C", "#FEF2F2", "Text >= 4.5:1"),
    ("Light: Focus Ring / Active Border (#2563EB) on Surface (#FFFFFF)", "#2563EB", "#FFFFFF", "UI Border >= 3.0:1"),
    ("Light: Input Control Border (#7C8894) on Surface (#FFFFFF)", "#7C8894", "#FFFFFF", "UI Border >= 3.0:1"),

    ("Dark: Body Text (#E5E7EB) on Surface (#111827)", "#E5E7EB", "#111827", "Text >= 4.5:1"),
    ("Dark: Body Text (#E5E7EB) on Page (#0B1220)", "#E5E7EB", "#0B1220", "Text >= 4.5:1"),
    ("Dark: Muted Text (#9CA3AF) on Surface (#111827)", "#9CA3AF", "#111827", "Text >= 4.5:1"),
    ("Dark: Muted Text (#9CA3AF) on Page (#0B1220)", "#9CA3AF", "#0B1220", "Text >= 4.5:1"),
    ("Dark: Button Text (#FFFFFF) on Primary Accent (#2563EB)", "#FFFFFF", "#2563EB", "Text >= 4.5:1"),
    ("Dark: Primary Accent Text (#60A5FA) on Surface (#111827)", "#60A5FA", "#111827", "Text >= 4.5:1"),
    ("Dark: Verified/Green Text (#4ADE80) on Surface (#111827)", "#4ADE80", "#111827", "Text >= 4.5:1"),
    ("Dark: Verified/Green Text (#4ADE80) on Dark Green (#052E16)", "#4ADE80", "#052E16", "Text >= 4.5:1"),
    ("Dark: Review/Amber Text (#FBBF24) on Surface (#111827)", "#FBBF24", "#111827", "Text >= 4.5:1"),
    ("Dark: Review/Amber Text (#FBBF24) on Dark Amber (#451A03)", "#FBBF24", "#451A03", "Text >= 4.5:1"),
    ("Dark: Error/Red Text (#F87171) on Surface (#111827)", "#F87171", "#111827", "Text >= 4.5:1"),
    ("Dark: Error/Red Text (#F87171) on Dark Red (#450A0A)", "#F87171", "#450A0A", "Text >= 4.5:1"),
    ("Dark: Focus Ring / Active Border (#3B82F6) on Surface (#111827)", "#3B82F6", "#111827", "UI Border >= 3.0:1"),
    ("Dark: Input Control Border (#6B7280) on Surface (#111827)", "#6B7280", "#111827", "UI Border >= 3.0:1"),
]

print(f"{'Item':<64} | {'Ratio':<8} | {'Standard':<18} | {'Status'}")
print("-" * 102)
for label, c1, c2, std in pairs:
    ratio = cr(c1, c2)
    min_req = 4.5 if "4.5" in std else 3.0
    status = "PASS" if ratio >= min_req else "FAIL"
    print(f"{label:<64} | {ratio:.2f}:1  | {std:<18} | {status}")
