# 시각화와 HTML 보고서

`truthound.profiler.visualization`은 native SVG·Plotly·ECharts renderer와 section 기반
HTML 보고서 생성기를 제공합니다. SVG에는 추가 차트 패키지가 필요 없습니다. Plotly는
`truthound[profiler-visualization]`을 설치하며 지원 범위는 `plotly>=5.24,<7`입니다.

## 보고서 생성

```python
from truthound.profiler.visualization import (
    HTMLReportGenerator, ProfileData, ReportConfig, ReportTheme,
)
data = ProfileData(
    table_name="example", row_count=3, column_count=1,
    columns=[{"name": "amount", "inferred_type": "integer",
              "null_ratio": 0, "unique_ratio": 1}],
)
generator = HTMLReportGenerator(ReportConfig(renderer="svg", theme=ReportTheme.LIGHT))
html = generator.generate(data)
generator.save(data, "profile.html")
```

`ProfileDataConverter.from_dict(profile.to_dict())`는 native profile payload와 ISO
`profiled_at`/`timestamp`를 받습니다. `generate_report(data, title="", theme="light",
renderer="auto", config=None, output_path=None, **options)`는 `ProfileData` 또는 dict를
받아 HTML 또는 저장 경로를 반환합니다. config를 제공하면 간편 인자보다 우선합니다.

호출별 `generate(data, config=...)` override는 별도 renderer instance를 사용하며 실패해도
이후 호출 설정을 바꾸지 않습니다. 명시적으로 주입한 chart renderer는 override에서도
유지합니다. 표 이름을 HTML escape하고 chart JSON의 script 구분자를 escape합니다.

## 설정과 리소스

`ReportConfig`는 title/subtitle, theme, logo 경로/base64, TOC/timestamp/summary 플래그,
sections, custom CSS/JS, `embed_resources`, language, renderer를 제공합니다.
기본 section은 overview·data quality·column details·patterns·recommendations이며 theme은
light·dark·professional·minimal·colorful입니다. TOC는 실제 section anchor에 연결됩니다.

`renderer="auto"`는 Plotly가 설치되어 있으면 선택하고 아니면 SVG를 사용합니다.
기본 `embed_resources=True`는 설치된 Plotly의 실제 JavaScript를 넣고 외부 script URL을
추가하지 않습니다. 보고서가 수 MiB 커질 수 있으며 artifact 예산은 계속 적용됩니다.
ECharts를 포함하려면 factory가 `EChartsChartRenderer(javascript=...)`에 신뢰된 JavaScript
문자열을 제공해야 합니다. 포함할 리소스가 없으면 원격으로 바꾸지 않고 실패합니다.
CDN을 명시적으로 선택하려면 `embed_resources=False`와 `use_cdn=True` renderer를
사용합니다. 그 HTML을 열 때 외부 script host 접근이 필요하며 native 생성 자체는 URL을
가져오지 않습니다.

custom CSS/JS·renderer·section template·theme·리소스 URL은 신뢰된 애플리케이션 설정입니다.
데이터 escape는 운영자 코드를 sandbox하지 않습니다. 공개 보고서에 비밀이나 비공개 원본
기록을 넣지 마세요. 전체 profile은 값과 권고를 포함할 수 있고 공개 동의는 렌더링과 별개입니다.

## 차트와 section

```python
from truthound.profiler.visualization import SVGChartRenderer, ChartData, ChartConfig, ChartType
chart = SVGChartRenderer().render(
    ChartData(labels=["A", "B"], values=[10, 20], title="Counts"),
    ChartConfig(chart_type=ChartType.BAR, width=400, height=300),
)
```

선택 backend의 `renderer.supports_chart_type(type)`을 확인하세요. enum에 있다고 모든
renderer가 구현한 것은 아닙니다. Plotly·ECharts는 script에 안전한 JSON을 사용하고
label/value 쌍과 table 행 길이가 다르면 조용히 자르지 않고 실패합니다. HTML 생성과 실제
브라우저 실행은 다른 검증입니다. Plotly 6.9.0의 JavaScript를 포함한 보고서를 로컬 브라우저에서
확인했으며 모든 backend·버전 검증은 아닙니다.

`section_registry`는 신뢰된 `SectionRenderer`와 `CustomSectionRenderer` callback을
등록합니다. `ReportTemplate`은 template/CSS/JS 보조 기능이며 wire가 선택하는 template
실행기는 아닙니다.

## 내보내기

`HTMLReportGenerator.save()`, `ReportExporter.save_html()`, HTML/JSON/PDF의
`to_file()`, `to_pdf()`는 비공개 sibling 파일·원자 교체·파일/디렉터리 flush로
게시합니다. 렌더링은 교체 전에 완료합니다. 교체 후 실패하면 새 내용이 남을 수
있으므로 재시도 전 목적지를 확인하세요. `render_pdf(profile, config=None)`와
`to_pdf_bytes(html)`는 경로에 쓰지 않고 native PDF bytes를 반환합니다.

`truthound[pdf]`와 WeasyPrint native 라이브러리가 필요합니다. WeasyPrint import가
불가능할 때만 기존 pdfkit fallback을 사용하며 native 렌더링 오류에서는 provider를
바꾸지 않습니다. macOS 호출 프로세스는 설치한 native 라이브러리 디렉터리를
`DYLD_FALLBACK_LIBRARY_PATH`에 지정해야 할 수 있습니다(Apple Silicon Homebrew는
`/opt/homebrew/lib`). 격리된 MCP factory도 자체 native 라이브러리 검색 설정을
소유합니다. Core는 시스템 라이브러리 경로를 바꾸지 않습니다.

프로파일 PDF는 HTML 설정이 Plotly여도 자동으로 정적 SVG를 선택합니다. SVG에
명시적 크기를 부여하고 A4 본문 폭에 맞추며 JavaScript 차트를 실행하지 않습니다.
직접 `to_pdf_bytes(html)`에 전달한 HTML의 리소스와 native provider 동작은 호출자가
소유합니다. SVG 막대 차트는 양수·음수를 지원합니다. renderer는 label/value 길이
불일치·비유한 값·음수 pie 값을 조용히 누락하거나 잘못 그리지 않고 거부합니다.

## 승인된 MCP 시각화

비공개 `ProfilerRuntime.visualizers`는 native `HTMLReportGenerator`를,
`visualization_slots`는 비공개 출력 경로를 별칭에 연결합니다. `inspect`로 별칭을 봅니다.
`visualization_export(profile_id, visualizer_id)`는 완료된 native profile을 요구하며 owner
예산 아래 실제 `text/html` artifact를 반환합니다.
`visualization_save(profile_id, visualizer_id, visualization_slot_id)`는 게시 전 크기를
검사하고 선택 슬롯에 저장하여 경로 없이 바이트 수·SHA-256을 반환합니다. 두 작업에는
정확한 승인·민감 결과 동의가 필요하며 wire로 CSS/JS·경로·renderer import를 받지 않습니다.
리소스·CDN 선택은 factory 책임입니다. 설치 worker에서 SVG artifact 읽기와 실제 저장
바이트 hash와 실제 PDF artifact를 검증합니다. `export_format="pdf"`를 지정하면
`application/pdf`를 내보내며 기본값은 HTML입니다. `visualization_save`에도 같은
format을 지정할 수 있습니다. 고정 저장 슬롯은 확장자가 아닌 선택 format의
bytes를 저장합니다.

## 다음 단계

- [ML 추론](ml-inference.md)
- [품질 점수](quality-scoring.md)
