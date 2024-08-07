
grp_chat_path_1 = "data/WhatsApp Chat with RIP VPN Meta World Peace/WhatsApp Chat with RIP VPN Meta World Peace.txt"
grp_chat_path_2 = "data/WhatsApp Chat with RIP Gajeebo_ Riyal in Jageebo/WhatsApp Chat with RIP Gajeebo Riyal in Jageebo.txt"

file1  = open(grp_chat_path_1,"r",encoding="utf8")
file2 = open(grp_chat_path_2,"r",encoding="utf8")

def process_line(line:str):
    no_of_colons = line.count(":")
    if no_of_colons <= 1:
        return None
    else:
        dialog = ''.join(line.split(":")[2:])
        return dialog


file_output = open("data/word2vec_data.txt","w",encoding="utf8")

i = 0
out_lines = []
for line in file1.readlines():
    dialog = process_line(line)
    if dialog is not None:
        out_lines.append(dialog)

for line in file2.readlines():
    dialog = process_line(line)
    if dialog is not None:
        out_lines.append(dialog)

file_output.writelines(out_lines)