import json
import subprocess

def generate_representations(group_generators, output_file, path_to_gap="/home/dagrawa2/gap/gap-4.11.1/bin/gap.sh", path_to_gapscript="libcode/gapcode.g"):
	input = "input := {};; output_file := \"{}\";; Read(\"{}\");; quit;".format( \
		str(group_generators), output_file, path_to_gapscript)
	sub = subprocess.Popen([path_to_gap, "-b"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
	output, errors = sub.communicate(input=input)
	sub.wait()
	assert "error" not in errors, errors
	with open(output_file, "r") as f:
		output_dict = json.load(f)
	return output_dict
