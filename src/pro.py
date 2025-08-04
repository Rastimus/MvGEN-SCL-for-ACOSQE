def classify_and_count_patterns(quadruples):
    """
    统计四种模式的四元组数量：
    EAEO: A不为NULL, O不为NULL
    EAIO: A不为NULL, O为NULL
    IAEO: A为NULL, O不为NULL
    IAIO: A为NULL, O为NULL
    """
    patterns = {
        'EAEO': [],
        'EAIO': [],
        'IAEO': [],
        'IAIO': []
    }
    
    for quad in quadruples:
        # 检查四元组格式是否正确
        if len(quad) != 4:
            continue
            
        aspect_term = quad[0]  # A
        opinion_term = quad[3]  # O
        
        # 判断模式
        if aspect_term != "NULL":
            if opinion_term != "NULL":
                patterns['EAEO'].append(quad)
            else:
                patterns['EAIO'].append(quad)
        else:
            if opinion_term != "NULL":
                patterns['IAEO'].append(quad)
            else:
                patterns['IAIO'].append(quad)
    
    # 统计结果
    counts = {pattern: len(quads) for pattern, quads in patterns.items()}
    
    return counts, patterns

def print_statistics(counts):
    """
    打印统计结果
    """
    total = sum(counts.values())
    print("\n=== Pattern Statistics ===")
    print(f"Total quadruples: {total}")
    for pattern, count in counts.items():
        percentage = (count / total) * 100 if total > 0 else 0
        print(f"{pattern}: {count} ({percentage:.2f}%)")

def process_and_analyze_file(input_file):
    # 读取四元组
    quadruples = []
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                # 假设每行是一个四元组列表的字符串形式
                try:
                    quad = eval(line.strip())
                    if isinstance(quad, list) and len(quad) == 4:
                        quadruples.append(quad)
                except:
                    continue
    
        # 统计模式
        counts, patterns = classify_and_count_patterns(quadruples)
        
        # 打印统计结果
        print_statistics(counts)
        
        return counts, patterns
        
    except Exception as e:
        print(f"Error processing file: {e}")
        return None, None

# 使用示例
def main():
    input_file = "C:/Users/26469/Desktop/代码/mvp/data/acos/laptop16/output.txt"  # 替换为实际的文件路径
    counts, patterns = process_and_analyze_file(input_file)
    
    if counts:
        # 可以进行更详细的分析
        # 例如，打印每种模式的具体例子
        print("\n=== Example quadruples for each pattern ===")
        for pattern, quads in patterns.items():
            if quads:
                print(f"\n{pattern} example:")
                print(quads[0])  # 打印每种模式的第一个例子

if __name__ == "__main__":
    main()
