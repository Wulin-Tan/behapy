---
name: "configure-mirrors"
description: "Configures pip and conda to use Tsinghua mirrors. Invoke when user needs to speed up package installation or asks to set up mirrors."
---

# Configure Mirrors

This skill helps configure `pip` and `conda` to use Tsinghua University mirrors.

## Usage

### pip
To use the mirror temporarily:
```bash
pip install -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple <package-name>
```

To configure global usage:
```bash
pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
```

### conda
Add the following to your `~/.condarc` file:

```yaml
channels:
  - defaults
show_channel_urls: true
default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
```
