# new_seedlings_talking_head

安装依赖
```
pip install gradio tqdm ffmpeg-python
```
运行文件
```
python gradio_new.py
```
将会看到以下类似的网址
```
Running on local URL:  http://127.0.0.1:7860
```
打开浏览器访问即可，若在远程服务器，则进行端口转发
本地电脑运行
```
ssh -L 7860:localhost:7860 username@your.server.ip
```
把 username 改成远程用户名
把 your.server.ip 改成你服务器 IP
然后访问你本地浏览器的：
http://localhost:7860
