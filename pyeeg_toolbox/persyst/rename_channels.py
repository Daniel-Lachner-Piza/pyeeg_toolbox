# import glob, os

# os.chdir("F:/FREIBURG_Simultaneous_OneHrFiles/pat_FR_375")
# lay_files_ls = []
# for file in glob.glob("*.lay"):
#     lay_files_ls.append(file)


# content_buffer = []
# channs_to_change = [1,2,5,6,7,8]
# for fpath in lay_files_ls:
#     with open(fpath, "r") as f:
#         # Read each line in the file
#         for line in f:
#             for ch_nr in channs_to_change:
#                 if f"P{ch_nr}-Ref=" in line and line[0]=='P':
#                     line = line.replace(f"P{ch_nr}-Ref=", f"PP{ch_nr}-Ref=")
#                     break
#             content_buffer.append(line)

#     with open(fpath, "w") as f:
#         f.writelines(content_buffer)
