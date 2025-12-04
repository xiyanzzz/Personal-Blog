---
title: Typora图床以及常规设置
date: 2023-12-29 17:01:30
tags: [其他, 工具]
categories: Others
---
# Typora 使用指南以及自定义设置

## 前言

在使用`Typora`之前，本人用的是另一款跨平台的开源的工具: `MarkText`，也完全够用，但如今来看，个人感觉只能算`Typora`的下位替代。`MarkText`使用会偶尔出现Bug，并且在我试图配置图像自动上传时遇到诸多不便。所以斥巨资(￥89！！！)买了`Typora`，用了一年以后的我不得不说，真香。

## 界面介绍

- Sidebar https://support.typora.io/File-Management/#open-folder

右上角：统计、outline大纲 、显示/隐藏侧边栏

左上（打开侧边栏）：可切换侧边栏显示：大纲/文件数

文件树显示时下方可创建、排序、搜索、最近打开等操作

侧边栏右上可对所有文档进行全局搜索

## 设置

### 基本设置

- 文件-自动保存

> 在macOS上，操作系统将为基于文档的应用程序（例如Typora或TextEdit）安排自动保存操作。因此，始终将“自动保存”作为系统功能启用。

- 编辑器
  - 打开匹配
  - 即时渲染
- 图像-见下问
  - 对本地、网络位置使用
  - 可以则优先相对
  - 自动转义（空格、中文）
- Markdown
  - 严格模式
  - 全打开扩展语法





## 图像上传（picgo-core + github）

- 说明 https://support.typora.io/Images/#when-insert-images

> In Markdown, images are written like `![alt](src)`. The `src` here can either be a URL or an absolute/relative file path.

语法加链接、本地拖动、粘贴板粘贴均可

> Typora supports paste image data from clipboard, **after telling typora where to put those images**. Typora would put image data into given folder or server, then insert images referring to that stored file or URL. Please

- picgo-core

```
1. 安装homebrew（mac）并添加至PATH
2. brew install node
3. npm install picgo -g
	以及安装插件 picgo install super-prefix
	/opt/homebrew/bin/
4. github创建repo，获取token
5. 配置picgo（~\.picgo\config.json）
6. 设置typora 注：上传服务选 自定义命令，详见参考文档
7. 验证
```

- 验证 （上述设置完需要重启一下，否则可能加载失败）

![截图粘贴图片成功上传](https://raw.githubusercontent.com/xiyanzzz/Picture-store/main/MarkText_pic/2023/12/29/20231229-113618.png)

## 自定义主题

### 基于Lapis的修改

https://support.typora.io/Add-Custom-CSS/

原主题展示

![image-20231229151738377](https://raw.githubusercontent.com/xiyanzzz/Picture-store/main/MarkText_pic/2023/12/29/20231229-151738.png)

想要的效果：(本人前端知识基本为零，只能靠猜和搜索一点点改)

- 白色主题下希望侧边栏能深一点的颜色区别（如Forest）
- Table希望行按深浅交叉颜色显示（如Mint）
- 二级标题希望能像默认主题一样有分割线（如Purple）

| Column1 | Column2 | Column3 |
| ------- | ------- | ------- |
| 1       | 1.1     | 1.2     |
| 2       | 2.1     | 2.2     |
| 3       | 3.1     | 3.2     |

```python
import numpy as np
a = 1
b = 2
print(a+b)
```



![Screenshot 2023-12-29 at 16.45.52](https://raw.githubusercontent.com/xiyanzzz/Picture-store/main/MarkText_pic/2023/12/29/20231229-164650.png)

```css
#write h(1~6) {
    font-family: 'SourceHanSerifCN'; \思源宋体
    padding: 0px; \内边距为0像素
    color: var(--primary-color); \页面预设颜色
}

#write h1 {
    text-align: center; \居中
    font-size: 2rem; \字体大小2rem
}

#write h2 {
    background-color: var(--header-span-color); \背景颜色
    color: var(--bg-color); \文字颜色
    padding: 1px 12.5px; \内边距
    border-radius: 4px; \圆角边框
    display: inline-block; \
}

#write h2 {
    font-size: 1.5rem; \字体大小
}

#write h2 {
    margin: .3em 0; \上下间距
}

#write h2 a {
    color: var(--bg-color);
    border-bottom-color: var(--bg-color) !important;
}

#write h2 strong {
    color: var(--bg-color);
}

#write h2 code {
    color: var(--bg-color);
    background-color: var(--header-span-color);
}


\\ 目标h2样式
h2 {
    position: relative;
    margin-top: 2rem;
    margin-bottom: 1rem;
    font-weight: var(--header-weight);
    line-height: 1.3;
    cursor: text;
    color: var(--title-color);
    font-family: var(--title-font);
    
    padding-left: 0.4em;
    border-left: 0.4em solid var(--title-color); \标题左侧边框
    border-bottom: 1px solid var(--title-color); \标题下界边框
}

h2 tt,
h2 code {
    font-size: inherit;
}

// 尝试修改目标h2
#write h2 {
    /*
    background-color: var(--header-span-color);
    color: var(--bg-color);
    padding: 1px 12.5px;
    border-radius: 4px;
    display: inline-block;
    */
    font-size: 1.5rem;
    /* 修改 */
    position: relative;
    margin-top: 2rem;
    margin-bottom: 1rem;
    font-weight: var(normal);
    line-height: 1.3;
    cursor: text;
    color: var(--primary-color);

    padding-left: 0.4em;
    border-left: 0.4em solid var(--header-span-color);
    border-bottom: 1px solid var(--header-span-color);
}


```

半成品效果

![效果图](https://raw.githubusercontent.com/xiyanzzz/Picture-store/main/MarkText_pic/2023/12/29/20231229-200254.png)

良好

![image-20231229200710842](https://raw.githubusercontent.com/xiyanzzz/Picture-store/main/MarkText_pic/2023/12/29/20231229-200711.png)

- 待解决问题

选中文件时希望能以黑色为背景高亮

## 图像加标题

- html居中

```html
<center><p>这里放标题</p></center>
```



- html整体封装

```html
<div>			<!--块级封装-->
    <center>	<!--将图片和文字居中-->
    <img src="图片路径放这里"
         alt="无法显示图片时显示的文字"
         style="zoom:这里写图片的缩放百分比"/>
    <br>		<!--换行-->
    这里是图片的标题	<!--标题-->
    </center>
</div>
————————————————
版权声明：本文为CSDN博主「Ladybug_」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/m0_47098916/article/details/117827687
```

