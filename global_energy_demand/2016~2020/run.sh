#!/bin/bash

# 全球能耗计算完整流程运行脚本
# 
# 功能概述：
# 本脚本用于依次运行全球能耗计算的完整流程，包括：
# 1. 数据加载和预处理 (11_load_data.py)
# 2. 网格点BAIT和能耗计算 (22_c_DD_pop.py)  
# 3. 国家级别能耗聚合 (33_country.py)
#
# 使用方法：
# ./run.sh [选项]
#
# 选项：
# -h, --help     显示帮助信息
# -s, --step     指定从哪个步骤开始 (1, 2, 3)
# -y, --years    指定要处理的年份，用逗号分隔 (如: 2016,2017,2018,2019,2020)
# -c, --check    仅检查依赖文件，不运行脚本
# -v, --verbose  显示详细输出
#
# 示例：
# ./run.sh                           # 运行所有步骤
# ./run.sh -s 2                      # 从步骤2开始
# ./run.sh -y 2019,2020              # 只处理2019和2020年
# ./run.sh -s 2 -y 2019              # 从步骤2开始，只处理2019年
# ./run.sh -c                        # 仅检查依赖文件

set -e  # 遇到错误时退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 默认配置
DEFAULT_YEARS="2016,2017,2018,2019,2020"
START_STEP=1
CHECK_ONLY=false
VERBOSE=false

# 脚本信息
declare -A SCRIPTS=(
    [1]="11_load_data.py|数据加载和预处理|加载2016-2020年气候数据并提取气象数据"
    [2]="22_c_DD_pop.py|网格点BAIT和能耗计算|计算各年份的BAIT和能耗数据"
    [3]="33_country.py|国家级别能耗聚合|进行国家级别能耗聚合"
)

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

# 显示帮助信息
show_help() {
    echo "全球能耗计算完整流程运行脚本"
    echo ""
    echo "使用方法:"
    echo "  $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -h, --help     显示此帮助信息"
    echo "  -s, --step     指定从哪个步骤开始 (1, 2, 3)"
    echo "  -y, --years    指定要处理的年份，用逗号分隔 (如: 2016,2017,2018,2019,2020)"
    echo "  -c, --check    仅检查依赖文件，不运行脚本"
    echo "  -v, --verbose  显示详细输出"
    echo ""
    echo "示例:"
    echo "  $0                           # 运行所有步骤"
    echo "  $0 -s 2                      # 从步骤2开始"
    echo "  $0 -y 2019,2020              # 只处理2019和2020年"
    echo "  $0 -s 2 -y 2019              # 从步骤2开始，只处理2019年"
    echo "  $0 -c                        # 仅检查依赖文件"
    echo ""
    echo "步骤说明:"
    echo "  1. 数据加载和预处理 - 加载2016-2020年气候数据并提取气象数据"
    echo "  2. 网格点BAIT和能耗计算 - 计算各年份的BAIT和能耗数据"
    echo "  3. 国家级别能耗聚合 - 进行国家级别能耗聚合"
}

# 解析命令行参数
parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -s|--step)
                START_STEP="$2"
                shift 2
                ;;
            -y|--years)
                YEARS="$2"
                shift 2
                ;;
            -c|--check)
                CHECK_ONLY=true
                shift
                ;;
            -v|--verbose)
                VERBOSE=true
                shift
                ;;
            *)
                log_error "未知选项: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

# 检查Python环境
check_python() {
    log_info "检查Python环境..."
    
    if ! command -v python &> /dev/null; then
        log_error "Python未安装或不在PATH中"
        exit 1
    fi
    
    PYTHON_VERSION=$(python --version 2>&1)
    log_success "Python环境检查通过: $PYTHON_VERSION"
}

# 检查依赖文件
check_dependencies() {
    log_info "检查依赖文件..."
    
    # 检查必要的输入文件
    local required_files=(
        "/z/local_environment_creation/Population/gpw-v4-population-count-adjusted-to-2015-unwpp-country-totals-rev11_2020_30_sec_tif/gpw_v4_population_count_adjusted_to_2015_unwpp_country_totals_rev11_2020_30_sec_aligned_to_MERRA2.tif"
        "/z/local_environment_creation/energy_consumption_gird/parameters.csv"
        "/z/local_environment_creation/energy_consumption_gird/result/point_country_mapping.csv"
        "/z/local_environment_creation/energy_consumption/2016-2020result/processed_countries.csv"
    )
    
    local missing_files=()
    for file_path in "${required_files[@]}"; do
        if [[ ! -f "$file_path" ]]; then
            missing_files+=("$file_path")
        fi
    done
    
    if [[ ${#missing_files[@]} -gt 0 ]]; then
        log_error "以下依赖文件不存在:"
        for file_path in "${missing_files[@]}"; do
            log_error "  $file_path"
        done
        return 1
    fi
    
    # 检查天气数据目录
    local weather_base_dir="/z/local_environment_creation/energy_consumption_gird/weather"
    if [[ ! -d "$weather_base_dir" ]]; then
        log_error "天气数据目录不存在: $weather_base_dir"
        return 1
    fi
    
    # 检查各年份的天气数据
    IFS=',' read -ra YEAR_ARRAY <<< "$YEARS"
    for year in "${YEAR_ARRAY[@]}"; do
        year_dir="$weather_base_dir/$year"
        if [[ ! -d "$year_dir" ]]; then
            log_warning "年份 $year 的天气数据目录不存在: $year_dir"
        else
            local slv_dir="$year_dir/M2T1NXSLV"
            local rad_dir="$year_dir/M2T1NXRAD"
            if [[ ! -d "$slv_dir" ]]; then
                log_warning "年份 $year 的SLV数据目录不存在: $slv_dir"
            fi
            if [[ ! -d "$rad_dir" ]]; then
                log_warning "年份 $year 的RAD数据目录不存在: $rad_dir"
            fi
        fi
    done
    
    log_success "依赖文件检查完成"
    return 0
}

# 创建输出目录
create_output_directories() {
    log_info "创建输出目录..."
    
    local output_dirs=(
        "/z/local_environment_creation/energy_consumption_gird/result/data"
        "/z/local_environment_creation/energy_consumption_gird/result/result_half"
        "/z/local_environment_creation/energy_consumption_gird/result/result"
    )
    
    for output_dir in "${output_dirs[@]}"; do
        if mkdir -p "$output_dir" 2>/dev/null; then
            log_success "创建目录: $output_dir"
        else
            log_error "创建目录失败: $output_dir"
            return 1
        fi
    done
    
    return 0
}

# 运行Python脚本
run_python_script() {
    local script_file="$1"
    local script_name="$2"
    local script_description="$3"
    
    log_info "开始运行: $script_name"
    log_info "描述: $script_description"
    log_info "脚本: $script_file"
    
    local start_time=$(date +%s)
    
    # 检查脚本文件是否存在
    if [[ ! -f "$script_file" ]]; then
        log_error "脚本文件不存在: $script_file"
        return 1
    fi
    
    # 运行脚本
    if [[ "$VERBOSE" == "true" ]]; then
        if python "$script_file"; then
            local end_time=$(date +%s)
            local duration=$((end_time - start_time))
            log_success "$script_name 执行成功 (耗时: ${duration}秒)"
            return 0
        else
            local end_time=$(date +%s)
            local duration=$((end_time - start_time))
            log_error "$script_name 执行失败 (耗时: ${duration}秒)"
            return 1
        fi
    else
        if python "$script_file" > /dev/null 2>&1; then
            local end_time=$(date +%s)
            local duration=$((end_time - start_time))
            log_success "$script_name 执行成功 (耗时: ${duration}秒)"
            return 0
        else
            local end_time=$(date +%s)
            local duration=$((end_time - start_time))
            log_error "$script_name 执行失败 (耗时: ${duration}秒)"
            return 1
        fi
    fi
}

# 主函数
main() {
    # 解析命令行参数
    parse_arguments "$@"
    
    # 设置默认年份
    if [[ -z "$YEARS" ]]; then
        YEARS="$DEFAULT_YEARS"
    fi
    
    # 验证步骤参数
    if [[ ! "$START_STEP" =~ ^[1-3]$ ]]; then
        log_error "无效的步骤参数: $START_STEP (必须是1, 2, 或3)"
        exit 1
    fi
    
    # 记录开始时间
    local start_time=$(date '+%Y-%m-%d %H:%M:%S')
    log_info "=========================================="
    log_info "全球能耗计算完整流程开始"
    log_info "开始时间: $start_time"
    log_info "处理年份: $YEARS"
    log_info "起始步骤: $START_STEP"
    log_info "=========================================="
    
    # 检查Python环境
    check_python
    
    # 检查依赖文件
    if ! check_dependencies; then
        log_error "依赖文件检查失败，请检查文件路径和权限"
        exit 1
    fi
    
    if [[ "$CHECK_ONLY" == "true" ]]; then
        log_info "仅检查模式，跳过脚本执行"
        exit 0
    fi
    
    # 创建输出目录
    if ! create_output_directories; then
        log_error "创建输出目录失败"
        exit 1
    fi
    
    # 获取当前脚本所在目录
    local script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    
    # 确定要运行的步骤
    local steps_to_run=()
    case $START_STEP in
        1) steps_to_run=(1 2 3) ;;
        2) steps_to_run=(2 3) ;;
        3) steps_to_run=(3) ;;
    esac
    
    # 运行脚本
    local success_count=0
    local total_steps=${#steps_to_run[@]}
    
    for step in "${steps_to_run[@]}"; do
        local script_info="${SCRIPTS[$step]}"
        IFS='|' read -r script_file script_name script_description <<< "$script_info"
        
        log_info "------------------------------------------"
        log_info "步骤 $step/$total_steps: $script_name"
        log_info "------------------------------------------"
        
        local script_path="$script_dir/$script_file"
        
        if run_python_script "$script_path" "$script_name" "$script_description"; then
            ((success_count++))
            log_success "步骤 $step 完成"
        else
            log_error "步骤 $step 失败"
            log_error "后续步骤将不会执行"
            break
        fi
    done
    
    # 记录结束时间
    local end_time=$(date '+%Y-%m-%d %H:%M:%S')
    log_info "=========================================="
    log_info "全球能耗计算完整流程结束"
    log_info "结束时间: $end_time"
    log_info "成功步骤: $success_count/$total_steps"
    
    if [[ $success_count -eq $total_steps ]]; then
        log_success "🎉 所有步骤执行成功！"
        exit 0
    else
        log_error "❌ 部分步骤执行失败"
        exit 1
    fi
}

# 运行主函数
main "$@"
