#!/usr/bin/env pwsh
# Кроссплатформенный скрипт запуска для Windows (PowerShell)
# Использование: .\run.ps1 <команда> [аргументы]

param(
    [Parameter(Position=0)]
    [string]$Command = "help",
    
    [Parameter(ValueFromRemainingArguments=$true)]
    [string[]]$Arguments
)

$Python = "python"
$DataDir = "data"
$ResultsDir = "results"
$AssetsDir = "assets"

# Создаем директории если их нет
if (!(Test-Path $DataDir)) { New-Item -ItemType Directory -Force -Path $DataDir | Out-Null }
if (!(Test-Path $ResultsDir)) { New-Item -ItemType Directory -Force -Path $ResultsDir | Out-Null }
if (!(Test-Path $AssetsDir)) { New-Item -ItemType Directory -Force -Path $AssetsDir | Out-Null }

function Run-Generate {
    param(
        [string]$Type = "blobs",
        [string]$Output = ""
    )
    if ($Output -eq "") { $Output = "$DataDir/${Type}.csv" }
    Write-Host "Генерация данных типа '$Type' в $Output..."
    & $Python tools/generate_data.py --type $Type --output $Output
}

function Run-Cluster {
    param(
        [string]$Algo = "dbscan",
        [string]$Input = "",
        [string]$Output = ""
    )
    if ($Input -eq "") { 
        Write-Error "Необходимо указать входной файл через --input"
        return 
    }
    if ($Output -eq "") { 
        $FileName = [System.IO.Path]::GetFileNameWithoutExtension($Input)
        $Output = "$ResultsDir/${Algo}_${FileName}.csv" 
    }
    
    Write-Host "Кластеризация алгоритмом '$Algo' файла $Input..."
    if ($Algo -eq "dbscan") {
        & $Python src/dbscan.py --input $Input --output $Output
    } elseif ($Algo -eq "kmeans") {
        & $Python src/kmeans.py --input $Input --output $Output
    } else {
        Write-Error "Неизвестный алгоритм: $Algo. Используйте 'dbscan' или 'kmeans'."
    }
}

function Run-Compare {
    Write-Host "Генерация сравнения K-means и DBSCAN..."
    & $Python tools/compare_kmeans_dbscan.py
}

function Show-Help {
    Write-Host @"
Использование: .\run.ps1 <команда> [параметры]

Команды:
  generate [--type <blobs|moons>] [--output <файл>]
             Генерирует набор данных. По умолчанию: blobs -> data/blobs.csv

  cluster --algo <dbscan|kmeans> --input <файл> [--output <файл>]
             Запускает кластеризацию. 

  compare
             Генерирует визуальное сравнение алгоритмов в assets/

  help
             Показать эту справку

Примеры:
  .\run.ps1 generate --type moons
  .\run.ps1 cluster --algo dbscan --input data/moons.csv
  .\run.ps1 compare
"@
}

# Парсинг аргументов вручную для гибкости
if ($Command -eq "generate") {
    $Type = "blobs"
    $Output = ""
    for ($i = 0; $i -lt $Arguments.Length; $i++) {
        if ($Arguments[$i] -eq "--type" -and $i + 1 -lt $Arguments.Length) {
            $Type = $Arguments[$i+1]
            $i++
        }
        if ($Arguments[$i] -eq "--output" -and $i + 1 -lt $Arguments.Length) {
            $Output = $Arguments[$i+1]
            $i++
        }
    }
    Run-Generate -Type $Type -Output $Output
}
elseif ($Command -eq "cluster") {
    $Algo = ""
    $Input = ""
    $Output = ""
    for ($i = 0; $i -lt $Arguments.Length; $i++) {
        if ($Arguments[$i] -eq "--algo" -and $i + 1 -lt $Arguments.Length) {
            $Algo = $Arguments[$i+1]
            $i++
        }
        if ($Arguments[$i] -eq "--input" -and $i + 1 -lt $Arguments.Length) {
            $Input = $Arguments[$i+1]
            $i++
        }
        if ($Arguments[$i] -eq "--output" -and $i + 1 -lt $Arguments.Length) {
            $Output = $Arguments[$i+1]
            $i++
        }
    }
    if ($Algo -eq "") { $Algo = "dbscan" } # Default algo
    Run-Cluster -Algo $Algo -Input $Input -Output $Output
}
elseif ($Command -eq "compare") {
    Run-Compare
}
else {
    Show-Help
}
