# xpath 是在HTML文档中搜索内容的一门语言
# html是xml的一个子集
html = """
        <head>
            <meta charset='utf-8'>
            <title>Title</title>
        </head>
        
        <body>
            <author>
                <nick>周大强</nick>
                <nick>周芷若</nick>
                <nick class="name1">彩依伦</nick>
                <nick id="5">Bob</nick>
                <nick id="8">华强</nick>
                <div>
                    <nick>刚子</nick>
                </div>
                <span>
                    <nick>顺子</nick>
                </span>
            </author>
        </body>
    """

# xpath靠着节点关系找信息，如/book/name,或者book/author/nick
# 使用
from lxml import etree
tree = etree.XML(html)
# result_1 = tree.xpath("/book/author/nick/text()")
# result_2 = tree.xpath("/book/author/div/nick/text()")
# result_3 = tree.xpath("/book/author//nick/text()") # 拿到后代所有信息
# result_4 = tree.xpath("/book/author/*/nick/text()") # author 后面任意子解点，然后再寻找子解点中的nick节点
# result_5 = tree.xpath("./xxx/xxx")  #相对查找路径
# result_6 = tree.xpath("./xxx/xxx/@href") 查找属性对应的值
print(result)
