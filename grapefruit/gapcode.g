### functions

SortedElements := function(G)
	local generators, powers, elements;;
	generators := GeneratorsOfGroup(G);;
	powers := List(Reversed(generators), g->List([0..Order(g)-1], n->g^n));;
	elements := Unique(List(Cartesian(powers), L->Product(L)));;
	return elements;;
end;;

MyRightTransversal := function(G, H, sorted_elements)
	local out, sort_func, SortedMinimum;;
	sort_func := function(g, h)
		return Position(sorted_elements, g) < Position(sorted_elements, h);;
	end;;
	SortedMinimum := L -> sorted_elements[Minimum(List(L, g->Position(sorted_elements, g)))];;
	out := rec();;
	out.gaptransversal := RightTransversal(G, H);;
	out.elements := List(out.gaptransversal, g->SortedMinimum(List(H*g)));;
	out.perm := Sortex(out.elements, sort_func);;
	return out;;
end;;

MyPositionCanonical := function(transversal, g)
	local pos;;
	pos := PositionCanonical(transversal.gaptransversal, g)^transversal.perm;;
	return pos;;
end;;

SubgroupsUpToConjugacy := function(G, sorted_elements)
	local classes, reps, subgroups;;
	classes := ConjugacyClassesSubgroups(G);;
	reps := [];;
	for class in classes do
		subgroups := List(class);;
	pos := PositionMinimum(List(subgroups, H->Maximum(List(H, h->Position(sorted_elements, h)))));;
	if Order(G)=12 and Order(subgroups[pos])=2 and (2, 6)(3, 5) in subgroups[pos] then
		Add(reps, ConjugateSubgroup(subgroups[pos], (1, 2, 3, 4, 5, 6)));;
	else
		Add(reps, subgroups[pos]);;
	fi;;
	od;;
	return reps;;
end;;

LowIndexSubgroupsUpToParentNormalcy := function(H, transversal, index)
	local reps, transversal_cap_NH, subgroups;;
	reps := LowIndexSubgroups(H, index);;
	if Length(reps)=1 then
		return reps;;
	else
		reps := Filtered(reps, K->Order(K)<Order(H));;
		transversal_cap_NH := Filtered(transversal.elements, g->ConjugateSubgroup(H, g)=H);;
		subgroups := Set(Concatenation( List(reps, K->List(transversal_cap_NH, g->ConjugateSubgroup(K, g))) ));;
		Add(subgroups, H);;
		subgroups := Reversed(subgroups);;
		return subgroups;;
	fi;;
end;;

PermAndCocycle := function(transversal, g)
	local pc, t, tg, pos;;
	pc := rec(perm:=[], cocycle:=[]);;
	for t in transversal.elements do
		tg := t*g;;
		pos := MyPositionCanonical(transversal, tg);;
		Add(pc.perm, pos);;
		Add(pc.cocycle, tg*Inverse(transversal.elements[pos]));;
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

degree := Length(input[1]);;
generators := List(input, PermList);;
G := GroupWithGenerators(generators);;

sorted_elements := SortedElements(G);;

subgroups := SubgroupsUpToConjugacy(G, sorted_elements);;
output := [];;
for H in subgroups do
	transversal := MyRightTransversal(G, H, sorted_elements);;
	record := rec(H:=PermsToLists(H, degree), transversal:=PermsToLists(transversal.elements, degree), out:=[]);;

	perms := [];;
	cocycles := [];;
	for g in generators do
		pc := PermAndCocycle(transversal, g);;
		Add(perms, pc.perm);;
		Add(cocycles, pc.cocycle);;
	od;;

	n := Index(G, H);;
	B_n := HyperoctahedralGroup(n);;

	H_subgroups := LowIndexSubgroupsUpToParentNormalcy(H, transversal, 2);;
	for K in H_subgroups do
		signatures := List(cocycles, cocycle->CocycleSigns(cocycle, K));;
		signed_perms := ListN(perms, signatures, HyperoctahedralElement);;
		rho_generators := List(signed_perms, perm->SignedList(perm, n));;
		Add(record.out, rec(K:=PermsToLists(K, degree), rho_generators:=rho_generators));;
	od;;

	Add(output, record);;
od;;

LoadPackage("json");;
file := OutputTextFile(output_file, false);;
GapToJsonStream(file, output);;
CloseStream(file);;
