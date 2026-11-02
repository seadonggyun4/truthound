# Security reporting

For suspected data-access, raw-value disclosure, package provenance or execution
vulnerabilities, email the public package maintainer at **seadonggyun4@gmail.com**.
Include affected package/plugin versions, OS/client versions and a minimal
synthetic reproduction. Do not send private datasets, credentials or raw PII.
Use GitHub private vulnerability reporting if it is available for this repository.
Do not post an exploitable issue publicly before coordinating a fix.

Maintenance is best effort for the latest published release. No response-time SLA
or security certification is promised. Reproduce fixes against the affected
package and permission boundary; release notes and a GitHub advisory, where
appropriate, should identify affected and fixed versions after coordination.

Installing a Local MCP plugin grants no directory access. The Local surface
returns bounded aggregate evidence. Host handling of file paths, column names
and aggregates remains outside this server's no-raw-value guarantee. The separate
Core profile has additional sensitive-result permissions and is not the Local
plugin's three-tool contract.
