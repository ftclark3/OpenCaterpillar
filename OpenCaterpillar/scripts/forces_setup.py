from openawsem.functionTerms import *
from OpenCaterpillar.functionTerms import *
from openawsem.helperFunctions.myFunctions import *

try:
    from openmm.unit import angstrom
    from openmm.unit import kilocalorie_per_mole
except ModuleNotFoundError:
    from simtk.unit import angstrom
    from simtk.unit import kilocalorie_per_mole

# Coluzza 2011 considers several temperatures, reported as kT
lambda coluzza_2011_hb_params kT: {
    'e_H': -3.1*kT,
    'nu': 2.0,
    'sigma': 0.2, # units of nm
}


def set_up_forces(cat, computeQ=False, submode=-1):
    # apply forces
    forces = [
        basicTerms.con_term(cat),
        basicTerms.chain_term(cat),
        basicTerms.chi_term(cat),
        basicTerms.excl_term(cat),
        basicTerms.rama_term(cat),
        basicTerms.rama_proline_term(cat),
        #basicTerms.rama_ssweight_term(cat, k_rama_ssweight=2*8.368),
        caterpillarTerms.caterpillar_contact_term(cat),
        caterpillarTerms.caterpillar_hb_term(cat, *coluzza_2011_hb_params(1), 2),
        #templateTerms.fragment_memory_term(
        #    cat, frag_file_list_file="./single_frags.mem", 
        #    npy_frag_table="./single_frags.npy", UseSavedFragTable=False,
        #    fm_well_width=0.01, frag_table_dr=0.001, 
        #    min_seq_sep=2),
        #debyeHuckelTerms.debye_huckel_term(cat, chargeFile="charge.txt"),
    ]
    if computeQ:
        forces.append(biasTerms.rg_term(cat))
        forces.append(biasTerms.q_value(cat, "crystal_structure-cleaned.pdb", forceGroup=1))
        forces.append(biasTerms.qc_value(cat, "crystal_structure-cleaned.pdb"))
        # forces.append(partial_q_value(cat, "crystal_structure-cleaned.pdb", residueIndexGroup=list(range(0, 15)), forceGroup=1))
    if submode == 0:
        additional_forces = [
            # contact_term(cat),
        ]
        forces += additional_forces
    return forces
