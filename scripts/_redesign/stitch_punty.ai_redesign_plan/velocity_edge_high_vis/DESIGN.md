# Design System Specification: High-Performance Light Mode

## 1. Overview & Creative North Star: "The Aerodynamic Gallery"
The Creative North Star for this design system is **"The Aerodynamic Gallery."** Much like a high-end automotive gallery located on a sun-drenched pit lane, the UI must feel expansive, breathable, and hyper-functional. We are moving away from the "dense cockpit" feel of dark mode into an editorial, high-contrast environment designed for instantaneous data ingestion under intense glare.

To break the "standard template" look, we utilize **Intentional Asymmetry**. Key data points should be offset from the center-line, and overlapping elements (typography bleeding over image containers) should be used to create a sense of forward motion. This system isn't just a container for information; it is a precision instrument.

---

## 2. Colors: High-Chroma Precision
Our palette is engineered for visibility. We leverage the starkness of `#f8f9fa` to make our brand accents—the signature hot pink and electric cyan—vibrate with intent.

### Core Palette
*   **Surface (Background):** `#f8f9fa` — A clean, technical off-white that reduces eye strain compared to pure `#ffffff`.
*   **Primary (Action):** `#b30069` — A calibrated version of our hot pink, deepened slightly to maintain a 4.5:1 contrast ratio against light surfaces.
*   **Secondary (Technical):** `#006876` — Our electric cyan, pulled toward a deeper teal for essential readability on status indicators.
*   **On-Surface (Text):** `#191c1d` — A deep charcoal, avoiding the "jitter" of pure black while providing maximum contrast.

### The "No-Line" Rule
**Prohibit 1px solid borders for sectioning.** To define high-level layout areas, use background shifts. 
*   *Example:* A sidebar should use `surface_container_low` (`#f3f4f5`) against a `surface` background. 
*   *The Goal:* A seamless, molded look that feels like a single piece of high-performance carbon fiber rather than a collection of boxes.

### Surface Hierarchy & Nesting
Treat the UI as a physical stack of technical sheets:
1.  **Base Layer:** `surface` (`#f8f9fa`)
2.  **Sectional Layer:** `surface_container` (`#edeeef`) for grouping related content.
3.  **Component Layer:** `surface_container_lowest` (`#ffffff`) for cards or interactive elements to provide a subtle "lift" against the gray-tinted sections.

---

## 3. Typography: The Editorial Engine
We use typography as a structural element. **Space Grotesk** is our voice: it is idiosyncratic, mechanical, and ultra-readable.

*   **Display (L/M/S):** `spaceGrotesk`. Use these for hero stats (e.g., Top Speed, Lap Times). Use tight letter-spacing (-0.02em) to give it a "compressed" high-speed feel.
*   **Headlines:** `spaceGrotesk`. These should be high-contrast (`on_surface`). Use intentional asymmetry—align headlines to the left while keeping body text slightly indented.
*   **Body (L/M/S):** `inter`. We switch to Inter for long-form data or descriptions to provide a neutral, highly legible counterpoint to the aggressive Display font.
*   **Labels:** `spaceGrotesk` (All Caps). Use for metadata and technical specs. It conveys authority and precision.

---

## 4. Elevation & Depth: Tonal Layering
In high-glare environments, traditional shadows often wash out. We use **Tonal Layering** supplemented by **Ambient Shadows**.

### The Layering Principle
Avoid shadows for most UI. Instead, place a `surface_container_lowest` (Pure White) card on top of a `surface_container` background. The slight shift in brightness is enough to define the boundary without creating visual clutter.

### Ambient Shadows & Glassmorphism
*   **Floating Elements:** For "over-the-UI" elements like modals or floating action buttons, use an **Ambient Shadow**: `on_surface` color at 6% opacity with a 32px blur and 16px Y-offset.
*   **The Glass Rule:** To maintain the "High-End" feel, use **Backdrop Blur** (20px) on navigation bars and overlays using a semi-transparent `surface_bright` (80% opacity). This allows the colors of the "track" (the content below) to bleed through, softening the interface.

---

## 5. Components: Precision Primitives

### Buttons
*   **Primary:** `primary` fill with `on_primary` text. Use `lg` (0.5rem) roundedness for a modern, "molded" look.
*   **Secondary:** `outline` border (using the **Ghost Border** rule: 15% opacity) with `primary` text. No fill.
*   **Hover State:** Transition to `primary_container` for a subtle, high-end "glow" effect.

### Cards & Lists
*   **The Divider Ban:** Strictly forbid `<hr>` or 1px dividers. Use vertical white space from the spacing scale (e.g., `spacing.6` or `2rem`) to separate list items.
*   **Data Cards:** Use `surface_container_low` with a subtle `primary` vertical accent line (4px width) on the left edge to denote "active" or "high-priority" data.

### Input Fields
*   **Style:** Minimalist. No bottom line or full border. Use a `surface_container_high` fill with a `md` (0.375rem) corner radius.
*   **Focus State:** A 2px `secondary` (Electric Cyan) "Ghost Border" at 40% opacity.

### The Mascot Integration
The mascot should not be a focal point but a "watermark" of quality. Place it in the background of headers or empty states using `surface_variant` at 15% opacity. It should feel etched into the interface, not floating on top of it.

---

## 6. Do’s and Don’ts

### Do:
*   **DO** use white space as a functional tool. If a screen feels crowded, increase the spacing to `spacing.10` or `12`.
*   **DO** use `Space Grotesk` for numbers. It is a highly "technical" looking font that reinforces the brand's data-driven nature.
*   **DO** ensure all CTA text meets WCAG AA standards against the light background.

### Don’t:
*   **DON’T** use 100% black text on pure white. It creates "vibration" that is painful in outdoor sunlight. Stick to our `on_surface` charcoal.
*   **DON’T** use rounded corners larger than `xl` (0.75rem). Anything "pill-shaped" (except for small chips) feels too consumer-grade and loses the professional "edge."
*   **DON’T** use drop shadows on text. If the text isn't readable, change the background color or weight; do not rely on shadows for legibility.