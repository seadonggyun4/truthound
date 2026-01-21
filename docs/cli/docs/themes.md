# truthound docs themes

List available report themes.

## Synopsis

```bash
truthound docs themes
```

## Arguments

None.

## Options

None.

## Description

The `docs themes` command displays all available themes for report generation:

1. **Lists** all built-in themes
2. **Shows** theme descriptions
3. **Indicates** default theme

## Examples

### List Themes

```bash
truthound docs themes
```

Output:
```
Available report themes:

  light          - Clean and bright, suitable for most use cases
  dark           - Dark mode with vibrant colors, easy on the eyes
  professional   - Corporate style, subdued colors (default)
  minimal        - Minimalist design with monochrome accents
  modern         - Contemporary design with vibrant gradients
```

## Available Themes

### Light (`light`)

Clean, bright style suitable for most use cases.

**Characteristics:**
- White background
- Dark text
- Subtle shadows
- Color-coded charts

**Best for:**
- General documentation
- Printing
- Daytime viewing

```bash
truthound docs generate profile.json -o report.html --theme light
```

### Dark (`dark`)

Dark mode theme for reduced eye strain.

**Characteristics:**
- Dark background (#1e1e1e)
- Light text
- Reduced contrast
- Muted chart colors

**Best for:**
- Night viewing
- Presentations in dark rooms
- Developer preferences

```bash
truthound docs generate profile.json -o report.html --theme dark
```

### Professional (`professional`)

Corporate style with calm, muted colors. **This is the default theme.**

**Characteristics:**
- Neutral color palette
- Clean typography
- Subtle branding areas
- Business-appropriate styling

**Best for:**
- Business reports
- Stakeholder presentations
- Enterprise documentation

```bash
truthound docs generate profile.json -o report.html --theme professional
```

### Minimal (`minimal`)

Minimalist design with monochrome accents.

**Characteristics:**
- Maximum whitespace
- Grayscale with single accent color
- Simple charts
- Focus on content

**Best for:**
- Technical documentation
- Simple reports
- Print optimization

```bash
truthound docs generate profile.json -o report.html --theme minimal
```

### Modern (`modern`)

Contemporary design with vibrant gradients.

**Characteristics:**
- Gradient backgrounds
- Bold colors
- Dynamic styling
- Modern typography

**Best for:**
- Marketing presentations
- Demo reports
- External stakeholders

```bash
truthound docs generate profile.json -o report.html --theme modern
```

## Theme Comparison

| Theme | Background | Best For | File Size |
|-------|------------|----------|-----------|
| light | White | General use, printing | Small |
| dark | Dark gray | Night viewing | Small |
| professional | Light gray | Business reports | Small |
| minimal | White | Technical docs | Smallest |
| modern | Gradients | Marketing, demos | Medium |

## Use Cases

### 1. Generate Reports in All Themes

```bash
for theme in light dark professional minimal modern; do
  truthound docs generate profile.json \
    -o "report_${theme}.html" \
    --theme $theme
done
```

### 2. Theme Selection by Audience

```bash
# For executives
truthound docs generate profile.json -o exec_report.html --theme professional

# For developers
truthound docs generate profile.json -o dev_report.html --theme dark

# For marketing
truthound docs generate profile.json -o marketing_report.html --theme modern
```

### 3. Print vs. Screen

```bash
# For printing
truthound docs generate profile.json -o print_report.html --theme minimal

# For screen viewing
truthound docs generate profile.json -o screen_report.html --theme dark
```

## Exit Codes

| Code | Condition |
|------|-----------|
| 0 | Success |

## Related Commands

- [`docs generate`](generate.md) - Generate reports with themes

## See Also

- [Data Docs Guide](../../guides/datadocs.md)
- [White-labeling](../../concepts/advanced.md)
