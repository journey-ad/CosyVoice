# 命令行参数
param (
    [switch]$MakeZip = $false,
    [switch]$NoModel = $false,
    [switch]$NoEnv = $false
)

# 设置变量
$ProjectName = "CosyVoice2-Ex"
$CondaEnvName = "cosyvoice"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$TempDir = Join-Path $ProjectRoot ".build_temp"
$OutputDir = Join-Path $TempDir "${ProjectName}_Portable"
$EnvTarFile = Join-Path $TempDir "env.tar.gz"
$EnvHashFile = Join-Path $TempDir "env_hash.txt"

# 设置编码为UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

# 获取conda环境的包列表hash（排除注释行）
function Get-CondaEnvHash {
    param (
        [string]$EnvName
    )
    $packageList = (conda list -n $EnvName | Where-Object { $_ -notmatch '^\s*#' }) -join "`n"
    $stream = [System.IO.MemoryStream]::new([System.Text.Encoding]::UTF8.GetBytes($packageList))
    return (Get-FileHash -InputStream $stream -Algorithm SHA256).Hash
}

# 显示进度条
function Show-BuildProgress {
    param (
        [string]$Status,
        [int]$PercentComplete
    )
    Write-Progress -Activity "打包进度" -Status $Status -PercentComplete $PercentComplete
}

Write-Host "开始打包便携版..."
Show-BuildProgress -Status "初始化..." -PercentComplete 0


# 创建临时目录
if (-not (Test-Path $TempDir)) {
    New-Item -Path $TempDir -ItemType Directory | Out-Null
}

Show-BuildProgress -Status "准备输出目录..." -PercentComplete 10
# 创建输出目录
if (Test-Path $OutputDir) {
    Write-Host "清理旧的输出目录..."
    Remove-Item -Path $OutputDir -Recurse -Force
}
New-Item -Path $OutputDir -ItemType Directory | Out-Null
New-Item -Path (Join-Path $OutputDir "env") -ItemType Directory | Out-Null

# 检查conda环境是否需要更新
Show-BuildProgress -Status "检查conda环境..." -PercentComplete 20

if ($NoEnv) {
    Write-Host "已启用NoEnv，跳过复制conda环境..."
} else {
    $NeedUpdateEnv = $true
    if (Test-Path $EnvTarFile) {
        Write-Host "检查conda环境是否有更新..." -NoNewline
        $CurrentHash = Get-CondaEnvHash -EnvName $CondaEnvName
        if (Test-Path $EnvHashFile) {
            $OldHash = Get-Content $EnvHashFile
            if ($CurrentHash -eq $OldHash) {
                Write-Host "依赖未变动，使用缓存文件" -NoNewline
                $NeedUpdateEnv = $false
            }
        }
        Write-Host ""
    }

    # 导出或复制conda环境
    if ($NeedUpdateEnv) {
        Show-BuildProgress -Status "导出conda环境..." -PercentComplete 30
        Write-Host "正在导出conda环境..."
        conda pack -n $CondaEnvName -o $EnvTarFile
        # 保存环境hash
        Get-CondaEnvHash -EnvName $CondaEnvName > $EnvHashFile
    }

    Show-BuildProgress -Status "解压环境..." -PercentComplete 40
    Write-Host "正在解压环境..."
    tar -xzf $EnvTarFile -C (Join-Path $OutputDir "env")
}

# 设置需要排除的目录
$ExcludeDirs = @(
    "\.build_temp",
    "\.git",
    "__pycache__",
    "\.pytest_cache",
    "\.idea",
    ".*\.egg-info",
    "\.ipynb_checkpoints",
    "\.vscode",
    "\.history"
)

# 复制项目文件，排除指定目录
Show-BuildProgress -Status "复制项目文件..." -PercentComplete 60
Write-Host "正在复制项目文件..." -NoNewline

# 根据NoModel参数决定是否排除模型目录
if ($NoModel) {
    Write-Host "启用-NoModel参数，跳过复制模型文件..." -NoNewline
    $ExcludeDirs += "pretrained_models"
}
Write-Host ""

# 递归复制文件，排除指定目录
Get-ChildItem -Path $ProjectRoot -Recurse | 
    Where-Object { 
        $item = $_
        -not ($ExcludeDirs | Where-Object { $item.FullName -match [regex]::Escape($_) })
    } | 
    ForEach-Object {
        $targetPath = $_.FullName.Replace($ProjectRoot, $OutputDir)
        if ($_.PSIsContainer) {
            if (-not (Test-Path $targetPath)) {
                New-Item -Path $targetPath -ItemType Directory -Force | Out-Null
            }
        } else {
            $targetDir = Split-Path -Parent $targetPath
            if (-not (Test-Path $targetDir)) {
                New-Item -Path $targetDir -ItemType Directory -Force | Out-Null
            }
            Copy-Item -Path $_.FullName -Destination $targetPath -Force
        }
    }

# 创建启动脚本
Show-BuildProgress -Status "创建启动脚本..." -PercentComplete 70
Write-Host "创建启动脚本..."

# WebUI启动脚本
$webuiScript = @"
@echo off
>nul chcp 65001

title CosyVoice2-Ex WebUI
color 0f

echo.
<nul set /p="╔════════════════════════════════════════════════════════════════════════════════════╗" & echo.
<nul set /p="║                                                                                    ║" & echo.
<nul set /p="║   ██████╗  ██████╗ ███████╗██╗   ██╗██╗   ██╗ ██████╗ ██╗ ██████╗███████╗██████╗   ║" & echo.
<nul set /p="║  ██╔════╝ ██╔═══██╗██╔════╝╚██╗ ██╔╝██║   ██║██╔═══██╗██║██╔════╝██╔════╝╚════██╗  ║" & echo.
<nul set /p="║  ██║      ██║   ██║███████╗ ╚████╔╝ ██║   ██║██║   ██║██║██║     █████╗   █████╔╝  ║" & echo.
<nul set /p="║  ██║      ██║   ██║╚════██║  ╚██╔╝  ╚██╗ ██╔╝██║   ██║██║██║     ██╔══╝  ██╔═══╝   ║" & echo.
<nul set /p="║  ╚██████╗ ╚██████╔╝███████║   ██║    ╚████╔╝ ╚██████╔╝██║╚██████╗███████╗███████╗  ║" & echo.
<nul set /p="║   ╚═════╝  ╚═════╝ ╚══════╝   ╚═╝     ╚═══╝   ╚═════╝ ╚═╝ ╚═════╝╚══════╝╚══════╝  ║" & echo.
<nul set /p="║                                                                                    ║" & echo.
<nul set /p="╚════════════════════════════════════════════════════════════════════════════════════╝" & echo.
echo                            CosyVoice2-Ex WebUI 服务正在启动...
echo                        https://github.com/journey-ad/CosyVoice2-Ex
echo                               整合包制作: journey-ad
echo.

:: 设置基础路径
set BASE_DIR=%~dp0
set CONDA_ENV=%BASE_DIR%env

:: 激活conda环境
call "%CONDA_ENV%\Scripts\activate.bat"

:: 设置环境变量
set PYTHONPATH=%BASE_DIR%
set HF_HOME=%BASE_DIR%hf_download
set PATH=%CONDA_ENV%\Scripts;%CONDA_ENV%\Library\bin;%PATH%

:: 运行程序
echo 正在启动WebUI服务，请稍候...
python webui.py --port 8080 --open
pause
"@
Set-Content -Path (Join-Path $OutputDir "运行-CosyVoice2-Ex.bat") -Value $webuiScript -Encoding UTF8


# API服务启动脚本
$apiScript = @"
@echo off
>nul chcp 65001

title CosyVoice2-Ex API
color 1f

echo.
<nul set /p="╔════════════════════════════════════════════════════════════════════════════════════╗" & echo.
<nul set /p="║                                                                                    ║" & echo.
<nul set /p="║   ██████╗  ██████╗ ███████╗██╗   ██╗██╗   ██╗ ██████╗ ██╗ ██████╗███████╗██████╗   ║" & echo.
<nul set /p="║  ██╔════╝ ██╔═══██╗██╔════╝╚██╗ ██╔╝██║   ██║██╔═══██╗██║██╔════╝██╔════╝╚════██╗  ║" & echo.
<nul set /p="║  ██║      ██║   ██║███████╗ ╚████╔╝ ██║   ██║██║   ██║██║██║     █████╗   █████╔╝  ║" & echo.
<nul set /p="║  ██║      ██║   ██║╚════██║  ╚██╔╝  ╚██╗ ██╔╝██║   ██║██║██║     ██╔══╝  ██╔═══╝   ║" & echo.
<nul set /p="║  ╚██████╗ ╚██████╔╝███████║   ██║    ╚████╔╝ ╚██████╔╝██║╚██████╗███████╗███████╗  ║" & echo.
<nul set /p="║   ╚═════╝  ╚═════╝ ╚══════╝   ╚═╝     ╚═══╝   ╚═════╝ ╚═╝ ╚═════╝╚══════╝╚══════╝  ║" & echo.
<nul set /p="║                                                                                    ║" & echo.
<nul set /p="╚════════════════════════════════════════════════════════════════════════════════════╝" & echo.
echo                           CosyVoice2-Ex API 服务正在启动...
echo                      https://github.com/journey-ad/CosyVoice2-Ex
echo                             整合包制作: journey-ad
echo.

:: 设置基础路径
set BASE_DIR=%~dp0
set CONDA_ENV=%BASE_DIR%env

:: 激活conda环境
call "%CONDA_ENV%\Scripts\activate.bat"

:: 设置环境变量
set PYTHONPATH=%BASE_DIR%
set HF_HOME=%BASE_DIR%hf_download
set PATH=%CONDA_ENV%\Scripts;%CONDA_ENV%\Library\bin;%PATH%

:: 运行程序
echo 正在启动API服务，请稍候...
python api.py
pause
"@
Set-Content -Path (Join-Path $OutputDir "启动接口服务.bat") -Value $apiScript -Encoding UTF8

# 创建README
Show-BuildProgress -Status "创建README..." -PercentComplete 80
Write-Host "创建README文件..."
$readme = @"
CosyVoice2-Ex 便携版
===================

使用说明：
1. 运行-CosyVoice2-Ex.bat 启动WebUI界面
2. 启动接口服务.bat 启动API服务
3. 首次运行可能需要等待环境配置
4. 模型文件位于 pretrained_models 目录

注意事项：
- 请勿移动或删除env目录下的文件
- 如遇到问题，请检查环境变量设置
- 确保已下载所需的模型文件
"@
Set-Content -Path (Join-Path $OutputDir "便携版使用说明.txt") -Value $readme -Encoding UTF8

if ($MakeZip) {
    # 打包成zip
    Show-BuildProgress -Status "创建压缩包..." -PercentComplete 90
    Write-Host "正在创建最终压缩包..."
    $ZipFile = Join-Path $ProjectRoot "${ProjectName}_Portable.zip"
    Compress-Archive -Path (Join-Path $OutputDir "*") -DestinationPath $ZipFile -Force

    Show-BuildProgress -Status "完成" -PercentComplete 100
    Write-Host "打包完成！"
    Write-Host "输出文件：$ZipFile"
} else {
    Show-BuildProgress -Status "完成" -PercentComplete 100
    Write-Host "打包完成！"
    Write-Host "输出目录：$OutputDir"
}

pause
