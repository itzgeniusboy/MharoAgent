"""Colour themes.

Textual >= 0.86 has a first-class theme registry; older versions only take a
`primary` colour + CSS variables. `register_all()` works on both: on old
versions the variables are injected as an app-level CSS custom-property block.
"""

from __future__ import annotations

from typing import Any

# name -> theme definition. `vars` are exposed to CSS as $name.
THEMES: dict[str, dict[str, Any]] = {
    "mharo-night": {
        "description": "Default. Deep indigo, tuned for long sessions.",
        "primary": "#7c9bff",
        "secondary": "#a78bfa",
        "accent": "#22d3ee",
        "warning": "#f5b93c",
        "error": "#ff6b6b",
        "success": "#4ade80",
        "foreground": "#e6e9f2",
        "background": "#0b0d14",
        "surface": "#12151f",
        "panel": "#1a1e2c",
        "dark": True,
        "vars": {
            "muted": "#8b93a7",
            "dim": "#5b6379",
            "user-bg": "#171b28",
            "tool-bg": "#10131d",
            "diff-add": "#3fb950",
            "diff-del": "#f85149",
            "gauge-full": "#7c9bff",
            "gauge-empty": "#262c3f",
            "brand": "#7c9bff",
        },
    },
    "mharo-ink": {
        "description": "Near-black, high contrast, minimal chrome.",
        "primary": "#c9a227",
        "secondary": "#8ec07c",
        "accent": "#00b8a4",
        "warning": "#e78a4e",
        "error": "#ea6962",
        "success": "#a9c07c",
        "foreground": "#d8d5c9",
        "background": "#090806",
        "surface": "#101010",
        "panel": "#1a1a18",
        "dark": True,
        "vars": {
            "muted": "#8a8878",
            "dim": "#575647",
            "user-bg": "#141412",
            "tool-bg": "#0d0d0b",
            "diff-add": "#a9c07c",
            "diff-del": "#ea6962",
            "gauge-full": "#c9a227",
            "gauge-empty": "#262621",
            "brand": "#c9a227",
        },
    },
    "mharo-slate": {
        "description": "Cool grey-blue, terminal-native feel.",
        "primary": "#6ea8fe",
        "secondary": "#7dd3fc",
        "accent": "#34d399",
        "warning": "#fbbf24",
        "error": "#fb7185",
        "success": "#34d399",
        "foreground": "#dbe2ea",
        "background": "#11151a",
        "surface": "#171c23",
        "panel": "#1f2630",
        "dark": True,
        "vars": {
            "muted": "#8794a4",
            "dim": "#586476",
            "user-bg": "#1a212a",
            "tool-bg": "#141a21",
            "diff-add": "#34d399",
            "diff-del": "#fb7185",
            "gauge-full": "#6ea8fe",
            "gauge-empty": "#232c38",
            "brand": "#6ea8fe",
        },
    },
    "mharo-day": {
        "description": "Light theme for bright rooms and projectors.",
        "primary": "#2f5bd7",
        "secondary": "#7c3aed",
        "accent": "#0e7490",
        "warning": "#b45309",
        "error": "#be123c",
        "success": "#15803d",
        "foreground": "#14181f",
        "background": "#f7f8fb",
        "surface": "#eef0f6",
        "panel": "#e3e7f0",
        "dark": False,
        "vars": {
            "muted": "#5c6577",
            "dim": "#8a93a5",
            "user-bg": "#e8ecf7",
            "tool-bg": "#f0f2f8",
            "diff-add": "#15803d",
            "diff-del": "#be123c",
            "gauge-full": "#2f5bd7",
            "gauge-empty": "#d3d9e8",
            "brand": "#2f5bd7",
        },
    },
}

DEFAULT_THEME = "mharo-night"


def register_all(app) -> str:
    """Register every theme on `app`. Returns the active theme name."""
    names = set(getattr(app, "available_themes", set()) or set())
    registered: list[str] = []
    for name, spec in THEMES.items():
        if name in names:
            registered.append(name)
            continue
        theme_cls = _theme_class()
        if theme_cls is None:  # very old Textual: only CSS variables available
            continue
        try:
            app.register_theme(
                theme_cls(
                    name=name,
                    primary=spec["primary"],
                    secondary=spec["secondary"],
                    accent=spec["accent"],
                    warning=spec["warning"],
                    error=spec["error"],
                    success=spec["success"],
                    foreground=spec["foreground"],
                    background=spec["background"],
                    surface=spec["surface"],
                    panel=spec["panel"],
                    dark=spec["dark"],
                    variables=dict(spec["vars"]),
                )
            )
            registered.append(name)
        except Exception:
            # Signature drift between Textual versions — fall back to variables only.
            continue
    app.mharo_css_vars = dict(THEMES[DEFAULT_THEME]["vars"])  # type: ignore[attr-defined]
    active = DEFAULT_THEME if registered else DEFAULT_THEME
    try:
        app.theme = active
    except Exception:
        pass
    return active


def _theme_class():
    try:
        from textual.theme import Theme  # Textual >= 0.86

        return Theme
    except Exception:
        return None


def cycle(current: str | None) -> str:
    names = list(THEMES)
    if current not in names:
        return names[0]
    return names[(names.index(current) + 1) % len(names)]
