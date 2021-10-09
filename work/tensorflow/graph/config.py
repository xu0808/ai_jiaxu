#!/usr/bin/env python
# coding: utf-8
# 基础配置
import os
import sys

# 数据文件目录
data_dir = 'E:\\study\\rec\\data\\graph\\input'
wiki_dir = os.path.join(data_dir, 'wiki')
wiki_edge_file = os.path.join(wiki_dir, 'Wiki_edgelist.txt')

out_dir = os.path.join('graph/gat', 'out')