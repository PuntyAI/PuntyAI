# Design System: High-Octane Data Editorial

## 1. Overview & Creative North Star

### Creative North Star: "The Neon Quant"
The visual identity of this design system is built at the intersection of high-stakes adrenaline and cold, hard probability. It moves beyond the standard "betting dashboard" by adopting a high-end editorial feel—think *Bloomberg Terminal* meets *Cyberpunk* street racing. 

We break the "template" look through **intentional asymmetry** and **tonal depth**. Rather than using a rigid, repetitive grid, we use expansive white space (or "dark space") to let data breathe, punctuated by aggressive, high-contrast highlights. Overlapping elements and "floating" data modules create a sense of speed and layered complexity, suggesting a platform that is as sophisticated as the algorithms powering it.

---

## 2. Colors & Surface Architecture

The palette is rooted in deep charcoal and near-black surfaces, allowing the electric accents to vibrate with intensity.

### The "No-Line" Rule
**Explicit Instruction:** Use of 1px solid borders for sectioning is strictly prohibited. Physical boundaries must be defined solely through background color shifts or subtle tonal transitions. For example, a `surface-container-low` module should sit directly on a `surface` background to define its perimeter.

### Surface Hierarchy & Nesting
Treat the UI as a physical stack of semi-translucent materials.
*   **Base Layer (`surface` / `#131318`):** The canvas.
*   **Section Layer (`surface-container-low` / `#1b1b20`):** Used for large content groupings.
*   **Component Layer (`surface-container-highest` / `#35343a`):** Used for individual interactive cards or data modules.
*   **The Glass & Gradient Rule:** For floating modals or "Age Verification" overlays, use a backdrop-blur (12px-20px) combined with a semi-transparent `surface-variant`. Main CTAs should utilize a linear gradient from `primary` (#ffb0cc) to `primary-container` (#ff46a0) at a 135-degree angle to provide "visual soul."

---

## 3. Typography: Speed vs. Precision

Our typography system is a dual-threat approach: **Space Grotesk** for high-impact brand moments and **Inter** for rapid data consumption.

*   **Display & Headlines (Space Grotesk):** These should be bold and aggressive. Use `display-lg` for hero statements. The wide, architectural nature of Space Grotesk mirrors the "high-speed" vibe of the track.
*   **Titles & Body (Inter):** A clean, neutral sans-serif ensures that even the most complex betting odds remain perfectly legible. 
*   **Labels & Data (Inter):** Small-scale data points should use `label-md` with increased letter-spacing (0.05em) to maintain clarity in high-density data visualizations.

---

## 4. Elevation & Depth: Tonal Layering

We convey importance through "lift" rather than lines.

*   **The Layering Principle:** Depth is achieved by stacking surface tokens. A `surface-container-lowest` card placed on a `surface-container-low` section creates a natural "recessed" effect, ideal for input fields or secondary data.
*   **Ambient Shadows:** Floating elements (like the Age Verification modal) must use extra-diffused shadows. 
    *   *Spec:* `0px 24px 48px rgba(0, 0, 0, 0.4)`. 
    *   Avoid grey shadows; use a dark tint of the background to ensure the shadow feels like a natural occlusion of light.
*   **The "Ghost Border" Fallback:** If a container absolutely requires a border for accessibility, use a "Ghost Border": `outline-variant` at 15% opacity. This provides a "suggestion" of a boundary without cluttering the visual field.

---

## 5. Components

### Buttons
*   **Primary:** Gradient-fill (`primary` to `primary-container`). Sharp corners with `sm` (0.125rem) rounding. High-contrast white or `on-primary` text.
*   **Secondary:** Ghost style. `tertiary` (#3cd7ff) "Ghost Border" with `on-surface` text. No fill.
*   **Action:** For "Bet" or "Confirm," use `secondary` (#ffb59d) to signal a distinct transaction type.

### Data Visualization & Cards
*   **Data Cards:** No dividers. Use `surface-container-high` as the card background and `surface-container-lowest` for the internal data table header to create separation.
*   **Progress Bars / Odds Meters:** Use the `tertiary` cyan for positive trends and `secondary` orange for warnings or high-risk bets.

### Inputs & Selection
*   **Fields:** Background should be `surface-container-lowest`. On focus, the "Ghost Border" should transition to a 100% opaque `tertiary` cyan glow.
*   **Chips:** Use `xl` (0.75rem) rounding (pill-shape) to contrast against the sharp-edged cards. This makes interactive filters feel "touchable."

---

## 6. Do's and Don'ts

### Do
*   **Do** use asymmetrical layouts. A 2/3 column for data and a 1/3 column for "Quick Bets" creates a professional, editorial rhythm.
*   **Do** utilize `surface-bright` for hover states on dark cards to provide a subtle "glow" effect.
*   **Do** ensure all typography for odds and numbers uses tabular font features (monospacing) to prevent "jumping" UI during live price updates.

### Don't
*   **Don't** use 100% opaque, high-contrast borders between list items. Use vertical white space (`spacing-4` or `spacing-6`) instead.
*   **Don't** use "pure" black (#000000). Use the `surface` token (#131318) to maintain the "charcoal" premium feel.
*   **Don't** round corners beyond `md` (0.375rem) for primary structural elements. We want "sharp and fast," not "soft and bubbly."