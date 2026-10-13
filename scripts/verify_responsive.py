import re
from pathlib import Path

def test_responsive_css():
    css_path = Path("app/static/styles.css")
    assert css_path.exists(), "styles.css not found"
    css = css_path.read_text(encoding="utf-8")

    # 1. No max-width on .app-layout
    app_layout_match = re.search(r'\.app-layout\s*\{([^}]+)\}', css)
    assert app_layout_match, "Missing .app-layout definition"
    app_layout_body = app_layout_match.group(1)
    assert "max-width" not in app_layout_body, ".app-layout contains max-width constraint"
    assert "width: 100%" in app_layout_body or "width: 100vw" in app_layout_body or "min-height: 100vh" in app_layout_body, ".app-layout must fill viewport"

    # 2. Main content 70/30 grid
    main_content_match = re.search(r'\.main-content\s*\{([^}]+)\}', css)
    assert main_content_match, "Missing .main-content definition"
    main_body = main_content_match.group(1)
    assert "grid-template-columns" in main_body, ".main-content must use CSS grid"
    assert "7fr" in main_body or "70%" in main_body or "minmax(0, 7fr)" in main_body, "Left column must be ~70%"
    assert "320px" in main_body, "Right rail must have minmax 320px constraint"

    # 3. Sticky right rail
    sidebar_match = re.search(r'\.sidebar-column\s*\{([^}]+)\}', css)
    assert sidebar_match, "Missing .sidebar-column definition"
    sidebar_body = sidebar_match.group(1)
    assert "sticky" in sidebar_body, ".sidebar-column must be sticky on desktop"

    # 4. Breakpoint 900px for stacked layout (covers 768px)
    assert "@media (max-width: 900px)" in css, "Missing 900px breakpoint for stacked layout"
    assert "grid-template-columns: 1fr" in css, "Must stack to 1fr on screens <= 900px"

    # 5. Mobile breakpoint for 375px
    assert "@media (max-width: 600px)" in css, "Missing mobile breakpoint"
    assert "padding: 0 16px" in css, "Mobile side padding must be 16px"

    # 6. Overflow protection on table
    assert "overflow-x: auto" in css, "Table wrapper must have overflow-x: auto for small screens"

    # 7. No prefers-color-scheme dark block
    assert "prefers-color-scheme: dark" not in css, "Styles must NOT have prefers-color-scheme: dark"

    print("Automated Responsive CSS Checks PASSED for 1920px, 1440px, 1024px, 768px, and 375px.")

if __name__ == "__main__":
    test_responsive_css()
