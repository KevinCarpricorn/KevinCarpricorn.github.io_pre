---
title: Hexo d部署报错之spawn failed的解决方案
date: 2021-05-09 09:25:28
tags: Hexo
---

关于Hexo部署的时候报错导致无法推送到github估计是很多小伙伴第一次接触Hexo框架编写博客的常见问题, 下面介绍两种解决方案.

<img src="https://i.loli.net/2021/05/09/fsRDw1AS2VpO35o.png">

## 解决方案(一)

1. 在博客文件夹(通常是**\blog**)中删除时 **.deploy_git** 文件
2. 命令行(terminal)[不推荐使用**cmd**, 使用 **git bash** 等] 中输入 `git config --global core.autocrlf false`把git加入系统环境变量
3. 重新执行`hexo c` `hexo g`  `hexo d`

上Google百度一查大部分都是这种方法, xdm可以自己试试看万一成了呢. 但我下面推荐另一种可能的解决方案



## 解决方案(二)

1. 首先用文本编辑器(我使用的是Notepad++)打开博客文件夹(通常是**\blog**)中的 **_config.yml** 配置文件 

2. 修改配置文件中的**repo**

   ```
   # Deployment
   ## Docs: https://hexo.io/docs/one-command-deployment
   deploy:
     type: git
     repo:	https://github.com/YourName/YourName.github.io.git(不要使用这个)
     		git@github.com:YourName/YourName.github.io.git(用这个)
     branch: master
   ```

3. 重新执行`hexo c` `hexo g`  `hexo d`

这样就大功告成啦, 很简单吧, 继续写你的博客吧!





## reference

https://blog.zhheo.com/p/128998ac.html

https://blog.csdn.net/njc_sec/article/details/89021083

