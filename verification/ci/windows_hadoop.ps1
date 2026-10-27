param(
    [ValidateSet('Build', 'Verify')][string]$Mode = 'Verify',
    [Parameter(Mandatory = $true)][string]$SdkRoot
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$hadoopVersion = '3.5.0'
$hadoopRevision = 'dbcc7cd797100e6b32cd84f85b53a5193a5f9af0'
$hadoopPatch = '7609530caaa260261a6302a4fb20fa6a26c65110'
$expectedFiles = @('bin/hadoop.dll', 'bin/winutils.exe', 'LICENSE.txt', 'NOTICE.txt')

function Assert-NativeSdk {
    $manifest = Join-Path $SdkRoot 'provenance.json'
    foreach ($path in @($SdkRoot, (Join-Path $SdkRoot 'bin'), $manifest)) {
        if ((Get-Item -LiteralPath $path).Attributes -band [IO.FileAttributes]::ReparsePoint) {
            throw 'HADOOP_SDK_PATH_ALIAS'
        }
    }
    $proof = Get-Content -LiteralPath $manifest -Raw | ConvertFrom-Json -AsHashtable
    if ($proof.version -ne $hadoopVersion -or $proof.source_sha -ne $hadoopRevision -or
        $proof.compatibility_patch_sha -ne $hadoopPatch) {
        throw 'HADOOP_SDK_SOURCE_MISMATCH'
    }
    if (@(Compare-Object @($proof.files.Keys | Sort-Object) @($expectedFiles | Sort-Object)).Count) {
        throw 'HADOOP_SDK_INVENTORY_MISMATCH'
    }
    $actualFiles = @(Get-ChildItem -LiteralPath $SdkRoot -Recurse -File |
        ForEach-Object { [IO.Path]::GetRelativePath($SdkRoot, $_.FullName).Replace('\', '/') } | Sort-Object)
    $allowedFiles = @($expectedFiles + @('provenance.json') | Sort-Object)
    if (@(Compare-Object $actualFiles $allowedFiles).Count) { throw 'HADOOP_SDK_UNEXPECTED_FILES' }
    foreach ($relative in $expectedFiles) {
        $file = Get-Item -LiteralPath (Join-Path $SdkRoot $relative)
        if ($file.PSIsContainer -or ($file.Attributes -band [IO.FileAttributes]::ReparsePoint)) {
            throw 'HADOOP_SDK_FILE_ALIAS'
        }
        $digest = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($digest -ne $proof.files[$relative]) { throw 'HADOOP_SDK_HASH_MISMATCH' }
        if ($relative -match '\.(dll|exe)$') {
            $stream = [IO.File]::OpenRead($file.FullName)
            try {
                if ($stream.ReadByte() -ne 77 -or $stream.ReadByte() -ne 90) {
                    throw 'HADOOP_SDK_NOT_PE'
                }
                $reader = [IO.BinaryReader]::new($stream)
                $stream.Position = 60
                $peOffset = $reader.ReadInt32()
                $stream.Position = $peOffset
                if ($reader.ReadUInt32() -ne 17744 -or $reader.ReadUInt16() -ne 34404) {
                    throw 'HADOOP_SDK_NOT_X64_PE'
                }
            } finally { $stream.Dispose() }
        }
    }
    Write-Output "Verified official Hadoop $hadoopVersion native SDK source and file hashes"
}

if (Test-Path -LiteralPath (Join-Path $SdkRoot 'provenance.json')) {
    Assert-NativeSdk
    return
}
if ($Mode -ne 'Build') { throw 'HADOOP_SDK_PROVENANCE_MISSING' }
if (Test-Path -LiteralPath $SdkRoot) { throw 'HADOOP_SDK_UNCLAIMED_DIRECTORY' }
if ($env:GITHUB_RUN_ID -notmatch '^\d+$' -or $env:GITHUB_RUN_ATTEMPT -notmatch '^\d+$') {
    throw 'HADOOP_SDK_RUN_ID_INVALID'
}
$sdkSource = Join-Path $env:RUNNER_TEMP "truthound-hadoop-source-$($env:GITHUB_RUN_ID)-$($env:GITHUB_RUN_ATTEMPT)"
if (Test-Path -LiteralPath $sdkSource) { throw 'HADOOP_SDK_SOURCE_ALREADY_EXISTS' }
git init --quiet $sdkSource
if ($LASTEXITCODE) { throw 'HADOOP_SDK_GIT_INIT_FAILED' }
git -C $sdkSource config core.longpaths true
if ($LASTEXITCODE) { throw 'HADOOP_SDK_LONG_PATHS_FAILED' }
git -C $sdkSource remote add origin https://github.com/apache/hadoop.git
if ($LASTEXITCODE) { throw 'HADOOP_SDK_GIT_REMOTE_FAILED' }
git -C $sdkSource -c core.autocrlf=false fetch --depth=1 origin $hadoopRevision
if ($LASTEXITCODE) { throw 'HADOOP_SDK_FETCH_FAILED' }
git -C $sdkSource -c core.autocrlf=false checkout --detach FETCH_HEAD
if ($LASTEXITCODE) { throw 'HADOOP_SDK_CHECKOUT_FAILED' }
if ((git -C $sdkSource rev-parse HEAD).Trim() -ne $hadoopRevision) {
    throw 'HADOOP_SDK_CHECKOUT_MISMATCH'
}
# Apache HADOOP-19977 renames the internal helper that collides with SDK 26100.
git -C $sdkSource fetch --depth=2 origin $hadoopPatch
if ($LASTEXITCODE) { throw 'HADOOP_SDK_PATCH_FETCH_FAILED' }
$patchFile = Join-Path $sdkSource 'native-sdk-compatibility.patch'
git -C $sdkSource show --format= $hadoopPatch | Set-Content -LiteralPath $patchFile -Encoding utf8
if ($LASTEXITCODE) { throw 'HADOOP_SDK_PATCH_READ_FAILED' }
git -C $sdkSource apply --check $patchFile
if ($LASTEXITCODE) { throw 'HADOOP_SDK_PATCH_CHECK_FAILED' }
git -C $sdkSource apply $patchFile
if ($LASTEXITCODE) { throw 'HADOOP_SDK_PATCH_APPLY_FAILED' }
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
$visualStudio = (& $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath).Trim()
if (-not $visualStudio) { throw 'HADOOP_SDK_CPP_TOOLS_MISSING' }
$toolsets = @(Get-ChildItem -Path (Join-Path $visualStudio 'MSBuild/Microsoft/VC/*/Platforms/x64/PlatformToolsets/v*/Toolset.props') |
    ForEach-Object { $_.Directory.Name } | Where-Object { $_ -match '^v\d+$' } |
    Sort-Object -Unique -Descending)
if (-not $toolsets.Count) { throw 'HADOOP_SDK_TOOLSET_MISSING' }
$toolset = $toolsets[0]
$env:Platform = 'x64'
$sdkModule = Join-Path $sdkSource 'hadoop-common-project/hadoop-common'
$buildLog = Join-Path $env:RUNNER_TEMP 'truthound-windows-hadoop-build.log'
# Use Apache's native-win compile phase. No product tests are skipped or relaxed.
& mvn.cmd -B -ntp -f (Join-Path $sdkModule 'pom.xml') compile -Pnative-win "-Duse.platformToolsetVersion=$toolset" 2>&1 |
    Tee-Object -FilePath $buildLog
if ($LASTEXITCODE) { throw 'HADOOP_SDK_NATIVE_BUILD_FAILED' }
New-Item -ItemType Directory -Path (Join-Path $SdkRoot 'bin') | Out-Null
foreach ($name in @('hadoop.dll', 'winutils.exe')) {
    Copy-Item -LiteralPath (Join-Path $sdkModule "target/bin/$name") -Destination (Join-Path $SdkRoot "bin/$name")
}
foreach ($name in @('LICENSE.txt', 'NOTICE.txt')) {
    Copy-Item -LiteralPath (Join-Path $sdkSource $name) -Destination (Join-Path $SdkRoot $name)
}
$hashes = @{}
foreach ($relative in $expectedFiles) {
    $hashes[$relative] = (Get-FileHash -LiteralPath (Join-Path $SdkRoot $relative) -Algorithm SHA256).Hash.ToLowerInvariant()
}
@{ version = $hadoopVersion; source_sha = $hadoopRevision; compatibility_patch_sha = $hadoopPatch; toolset = $toolset;
   image_os = $env:ImageOS; image_version = $env:ImageVersion; files = $hashes } |
    ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $SdkRoot 'provenance.json') -Encoding utf8
Assert-NativeSdk
