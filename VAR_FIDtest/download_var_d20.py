#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import urllib.request
import os
import sys
from urllib.parse import urlparse

def download_with_progress(url, filename):
    """使用urllib下载文件并显示进度条"""
    def reporthook(block_num, block_size, total_size):
        if total_size > 0:
            percent = min(block_num * block_size * 100 / total_size, 100)
            downloaded = min(block_num * block_size, total_size)

            # 格式化文件大小
            def format_bytes(bytes_num):
                for unit in ['B', 'KB', 'MB', 'GB']:
                    if bytes_num < 1024:
                        return f"{bytes_num:.1f} {unit}"
                    bytes_num /= 1024
                return f"{bytes_num:.1f} TB"

            progress_bar = '█' * int(percent / 2) + '░' * (50 - int(percent / 2))
            sys.stdout.write(f'\r[{progress_bar}] {percent:.1f}% ({format_bytes(downloaded)}/{format_bytes(total_size)})')
            sys.stdout.flush()

    try:
        print(f"开始下载: {url}")
        print(f"保存为: {filename}")
        print("=" * 80)

        # 下载文件
        urllib.request.urlretrieve(url, filename, reporthook)
        print("\n" + "=" * 80)
        print("✅ 下载完成!")

        # 显示文件信息
        file_size = os.path.getsize(filename)
        print(f"文件大小: {file_size / 1024 / 1024:.1f} MB")
        print(f"文件位置: {os.path.abspath(filename)}")

    except Exception as e:
        print(f"\n❌ 下载失败: {e}")
        return False

    return True

def main():
    url = "https://huggingface.co/FoundationVision/var/resolve/main/var_d20.pth"
    filename = "var_d20.pth"

    # 检查文件是否已存在
    if os.path.exists(filename):
        response = input(f"文件 {filename} 已存在，是否覆盖? (y/N): ")
        if response.lower() != 'y':
            print("下载已取消")
            return

    success = download_with_progress(url, filename)

    if success:
        print("\n🎉 所有任务完成!")
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()