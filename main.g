### functions

SubgroupsUpToConjugacy := function(G)
	local classes, reps;;
	classes := ConjugacyClassesSubgroups(G);;
	reps := List(classes, Representative);;
	return reps;;
end;;

AllLowIndexSubgroups := function(G, index)
	local reps, subgroups;;
	reps := LowIndexSubgroups(G, index);;
	subgroups := Concatenation( List(reps, H->AsList(ConjugacyClassSubgroups(G, H))) );;
	return subgroups;;
end;;

PermAndCocycle := function(transversal, g)
	local pc, t, tg, pos;;
	pc := rec(perm:=[], cocycle:=[]);;
	for t in transversal do
		tg := t*g;;
		pos := PositionCanonical(transversal, tg);;
		Add(pc.perm, pos);;
		Add(pc.cocycle, tg*Inverse(transversal[pos]));;
	od;;
	return pc;;
end;;

CocycleSigns := function(cocycle, K)
	local signs, c;;
	signs := [];;
	for c in cocycle do
		if c in K then
			Add(signs, 1);;
		else
			Add(signs, -1);;
		fi;;
	od;;
	return signs;;
end;;

HyperoctahedralGroup := function(n)
	local cycle, transpos, reflect, G;;
	if n = 1 then
		G := Group((1, 2));;
	else
		cycle := [2..2*n+1];;
		cycle[n] := 1;;
		cycle[2*n] := n+1;;
		cycle := PermList(cycle);;
		transpos := (1, 2)(n+1, n+2);;
		reflect := (1, n+1);;
		G := Group(cycle, transpos, reflect);;
	fi;;
	return G;;
end;;

HyperoctahedralElement := function(perm, signs)
	local n, element;;
	n := Length(perm);;
	element := perm + n*(1-signs)/2;;
	Append(element, (element+n) mod (2*n));;
	if 0 in element then
		element[Position(element, 0)] := 2*n;;
	fi;;
	element := PermList(element);;
	return element;;
end;;

SignedList := function(perm, n)
	local signed_list, list_perm, i, l;;
	signed_list := [];;
	list_perm := ListPerm(perm, 2*n);;
	for i in [1..n] do
		l := list_perm[i];;
		if l > n then
			Add(signed_list, n-l);;
		else
			Add(signed_list, l);;
		fi;;
	od;;
	return signed_list;;
end;;

PermsToLists := function(perms, n)
	local lists;;
	lists := List(perms, perm->ListPerm(perm, n));;
	return lists;;
end;;


### main

rho := (1, 2, 3, 4, 5, 6);;
tau := (1, 6)(2, 5)(3, 4);;
degree := 6;;

G := Group(rho, tau);;
generators := GeneratorsOfGroup(G);;

subgroups := SubgroupsUpToConjugacy(G);;
output := [];;
for H in subgroups do
	transversal := RightTransversal(G, H);;
	record := rec(H:=PermsToLists(H, degree), transversal:=PermsToLists(transversal, degree), out:=[]);;

	perms := [];;
	cocycles := [];;
	for g in generators do
		pc := PermAndCocycle(transversal, g);;
		Add(perms, pc.perm);;
		Add(cocycles, pc.cocycle);;
	od;;

	n := Index(G, H);;
	B_n := HyperoctahedralGroup(n);;

	H_subgroups := AllLowIndexSubgroups(H, 2);;
	for K in H_subgroups do
		signatures := List(cocycles, cocycle->CocycleSigns(cocycle, K));;
		signed_perms := ListN(perms, signatures, HyperoctahedralElement);;
		centralizer_generators := GeneratorsOfGroup (Centralizer(B_n, GroupWithGenerators(signed_perms)) );;
		centralizer_generators := List(centralizer_generators, perm->SignedList(perm, n));;
		Add(record.out, rec(K:=PermsToLists(K, degree), centralizer_generators:=centralizer_generators));;
	od;;

	Add(output, record);;
od;;

LoadPackage("json");;
file := OutputTextFile("output.json", false);;
GapToJsonStream(file, output);;
CloseStream(file);;
