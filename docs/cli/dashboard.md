# truthound dashboard

Launch an interactive dashboard for data exploration.

## Synopsis

```bash
truthound dashboard [OPTIONS]
```

## Arguments

None.

## Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--profile` | `-p` | None | Profile file to load |
| `--port` | | `8080` | Server port |
| `--host` | | `localhost` | Server host |
| `--title` | `-t` | `Truthound Dashboard` | Dashboard title |
| `--debug` | | `false` | Enable debug mode |

## Description

The `dashboard` command launches an interactive web dashboard:

1. **Starts** local web server
2. **Loads** profile data (if provided)
3. **Serves** interactive UI
4. **Enables** real-time data exploration

### Dashboard Features

- **Interactive data exploration** - Browse columns, filter data
- **Column filtering and search** - Find specific fields quickly
- **Real-time quality metrics** - Live validation status
- **Pattern visualization** - Charts and graphs
- **Data comparison** - Side-by-side analysis

## Installation

The dashboard requires additional dependencies:

```bash
pip install truthound[dashboard]
```

## Examples

### Basic Dashboard

```bash
truthound dashboard --profile profile.json
```

Opens dashboard at `http://localhost:8080`.

### Load Profile Data

```bash
truthound dashboard --profile profile.json
```

### Custom Port and Host

```bash
truthound dashboard --port 3000 --host 0.0.0.0
```

Opens dashboard at `http://0.0.0.0:3000` (accessible from network).

### Custom Title

```bash
truthound dashboard --title "Customer Data Explorer" --profile profile.json
```

### Debug Mode

```bash
truthound dashboard --debug --profile profile.json
```

Enables debug mode for the dashboard server.

!!! note "Debug Mode"
    The `--debug` flag is passed to the dashboard server. Actual behavior
    depends on the underlying Reflex framework implementation.

### Complete Example

```bash
truthound dashboard \
  --profile profile.json \
  --port 8080 \
  --host localhost \
  --title "Production Data Dashboard" \
  --debug
```

## Workflow

### 1. Generate Profile First

```bash
# Create profile from data
truthound auto-profile data.csv -o profile.json --format json

# Launch dashboard with profile
truthound dashboard --profile profile.json
```

### 2. Explore Data

1. Open browser at `http://localhost:8080`
2. Browse column statistics
3. Filter by data type or quality
4. View distribution charts
5. Export findings

## Dashboard Sections

### Overview
- Dataset summary (rows, columns, size)
- Overall quality score
- Quick stats

### Columns
- Column list with search
- Data type indicators
- Null percentage bars
- Click for details

### Quality
- Validation results
- Issue breakdown by severity
- Pass/fail rates

### Charts
- Distribution histograms
- Correlation matrix
- Missing value heatmap

### Settings
- Theme toggle (light/dark)
- Export options
- Refresh controls

## Use Cases

### 1. Data Exploration

```bash
# Quick data exploration
truthound auto-profile new_dataset.csv -o profile.json --format json
truthound dashboard --profile profile.json
```

### 2. Team Collaboration

```bash
# Share on local network
truthound dashboard --profile profile.json --host 0.0.0.0 --port 8080
# Team members access via http://<your-ip>:8080
```

### 3. Development Workflow

```bash
# Debug mode for development
truthound dashboard --debug --profile profile.json
```

### 4. Presentation Mode

```bash
# Custom title for presentations
truthound dashboard \
  --profile quarterly_profile.json \
  --title "Q4 Data Quality Review"
```

## Configuration

### Command-Line Configuration

All configuration is done via command-line options:

```bash
truthound dashboard \
  --profile profile.json \
  --port 3000 \
  --host 0.0.0.0 \
  --title "My Dashboard"
```

!!! note "Profile Required"
    The `--profile` option is required for the dashboard to display data.
    Running without a profile will result in an error.

## Comparison: Dashboard vs. Static Reports

| Feature | Dashboard | docs generate |
|---------|-----------|---------------|
| Interactivity | Full | None |
| Server required | Yes | No |
| Offline viewing | No | Yes |
| Real-time updates | Yes | No |
| Sharing | URL | File |
| Best for | Exploration | Documentation |

### When to Use Dashboard

- Interactive data exploration
- Team collaboration
- Real-time monitoring
- Development workflow

### When to Use Static Reports

- CI/CD artifacts
- Email distribution
- Offline viewing
- Permanent documentation

## Exit Codes

| Code | Condition |
|------|-----------|
| 0 | Server stopped gracefully |
| 1 | Server error |
| 2 | Invalid arguments |

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+C` | Stop server |
| `r` | Refresh data (in browser) |
| `/` | Focus search |
| `?` | Show help |

## Related Commands

- [`docs generate`](docs/generate.md) - Generate static reports
- [`auto-profile`](profiler/auto-profile.md) - Generate profile data
- [`profile`](core/profile.md) - Basic profiling

## See Also

- [Dashboard Guide](../dashboard/index.md)
- [Data Docs Guide](../guides/datadocs.md)
