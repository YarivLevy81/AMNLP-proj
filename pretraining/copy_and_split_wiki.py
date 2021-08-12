# python3 copy_and_split_wiki.py /home/yandex/AMNLP2021/data/wiki/all /home/yandex/AMNLP2021/sehaik/wiki_split/ 1000000 0
import sys
import os

f_path = sys.argv[1]
save_dir = sys.argv[2]
size = int(sys.argv[3])
seek_byte = int(sys.argv[4])

fsize = os.path.getsize(f_path)
wiki_file = open(f_path, "r")
wiki_file.seek(seek_byte)
count = seek_byte // size

while True:
    lines = wiki_file.read(size)
    res_path = os.path.join(save_dir, f"file_{count}")

    with open(res_path, "w") as f:
        f.write(lines)
    print(f"saved {res_path}. seek_byte={seek_byte}")

    seek_byte += size
    if (seek_byte > fsize):
        break
    count += 1

print("done")
wiki_file.close()