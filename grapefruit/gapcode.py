import json
import subprocess

def generate_representations(group_generators, filename, path_to_gap="~/gap/gap-4.11.1/gap.sh", path_to_gapscript="grapefruit/gapcode.g"):
	input = "input := {};; output_file := {};; Read({});; quit;".format( \
		str(group_generators), filename, path_to_gapscript)
	sub = subprocess.Popen([path_to_gap, "-b"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
	output, errors = sub.communicate(input=input)
	sub.wait()
	with open(filename, "r") as fp:
		output_dict = json.load(fp)
	return output_dict
