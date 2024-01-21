import re

def config_overwrite(overwrite, config):
    overwrite_dict = {}
    for item in overwrite.split(','):
        k, v = item.split('=')
        overwrite_dict[k] = v
    for k, v in overwrite_dict.items():
        depth = k.count('.')
        node, key = get_node(config, depth, k)
        node[key] = trans_type(v)
        
def trans_type(value):
    """将字符串的内容转换成python类型变量，目前只能支持布尔类型变量、整型变量、浮点型变量。暂不支持列表。
    @param: value: string 要替换的key
    """
    if value.lower() in ['false', 'true']:
        return False if value.lower() == 'false' else True
    if re.compile(r'^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$').match(value):
        if value.isnumeric():
            return int(value)
        else:
            return float(value)
    return value
    

def get_node(item, depth, key):
    """递归得到字典深层的节点
    @param: item:Dict 字典
    @param: depth:int 深度，当深度=0时，直接取当前节点的值，不用往下再取
    @param: key: 键名，当前字节需要访问的键的名字。深度不等于0时，是很多个键组合而成的字符串。例如，key1.key2.key3。
    """
    if depth == 0:
        return item, key
    else:
        curr_key = key.split('.')[0]
        next_key = key[len(curr_key)+1:]
        return get_node(item[curr_key], depth-1, next_key)