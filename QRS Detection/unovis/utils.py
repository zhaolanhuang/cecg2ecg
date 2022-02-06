import numpy as np
import logging as log

def numpy_table_to_beautiful_string(arr, axis_0, axis_1, axis_0_name, axis_1_name, col_width=5, header=""):
	ax_0_len = len(axis_0)
	ax_1_len = len(axis_1)
	axis_name = axis_0_name + "\\" + axis_1_name
	lines = []

	col_name_pat = "| {:^" + str(col_width) + "} "
	row_name_pat = "{:" + str(len(axis_name)) + "}"
	data_pattern = "| {:" + str(col_width) + ".2f} "
	row_pattern = data_pattern*ax_1_len

	lines.append(header + "\n")
	lines.append(axis_name + (col_name_pat*ax_1_len).format(*axis_1) + "\n")
	lines.append("-"*len(axis_name) + ("|" + "-"*(col_width+2))*ax_1_len + "\n")
	for ax_0_idx, ax_0_val in enumerate(axis_0):
		lines.append(row_name_pat.format(ax_0_val) + row_pattern.format(*arr[ax_0_idx,:]) + "\n")

	rtrn = ""
	for line in lines:
		rtrn += line

	return rtrn