# same imports that we use in the openawsem functionTerms files
from openmm.app import *
from openmm import *
from openmm.unit import *
import numpy as np


# copied from openawsem hydrogenBondTerms.py
def find_chain_index(res, chain_starts, chain_ends):
    """
    Find the index of the chain that contains the residue with index `res`.
    """
    chain_index = [int(chain_start<=res and res<=chain_end) for chain_start,chain_end in zip(chain_starts,chain_ends)]
    assert sum(chain_index) == 1, f"res: {res}, chain_starts: {chain_starts}, chain_ends: {chain_ends}, list: {chain_index}"
    return chain_index.index(1)


def load_gammas(filename):
    onebody_data = []
    pairwise_data = []
    with open(filename, 'r') as f:
        previous_line_blank = True
        for counter, line in enumerate(f):
            if counter < 68:
                continue # i don't know what these numbers mean, but they don't look like gammas
            line = line.strip()
            if line:
                if previous_line_blank:
                    onebody_data.append(float(line))
                else:
                    pairwise_data.append(float(line))
                previous_line_blank = False 
            else:
                previous_line_blank = True 
    #breakpoint()
    #matrix21 = np.full((21,21), fill_value=np.nan)
    #matrix21[np.triu_indices(21)] = data
    #burial_gamma = matrix21[0,1:] # first row is burial gammas, but there are only 20, so element 0 seems not to contain any information
    #pairwise_gamma = matrix21[1:,1:] # only main diagonal and upper triangular are filled in; np.nan on the lower triangle
    burial_gamma  = np.array(onebody_data)
    pairwise_gamma = np.full((20,20), fill_value=np.nan)
    pairwise_gamma[np.triu_indices(20)] = pairwise_data
    return burial_gamma, pairwise_gamma
def caterpillar_contact_term(
        cat, 
        r_max = 1.2, # this is like the 'eta' that we use to calculate rho in 
        OMEGA = 21.0, # this is the critical number of neighbors for the burial potential
        #burial_kappa = 4.0 # burial_kappa not in ivan's model but needed for MD (or may be needed to achieve good efficiency); using same value as awsem
        k_contact=4.184, k_burial=4.184,
        parametersLocation=None, 
        forceGroup=22,
        gamma_file='aapot-pot.dat'):
    
    burial_gamma, pairwise_gamma = load_gammas(gamma_file)
    burial_gamma = np.zeros(burial_gamma.shape)

    letter_to_index = {
        'A':0,'C':1,'D':2,'E':3,'F':4,'G':5,
        'H':6,'I':7,'K':8,'L':9,'M':10,'N':11,
        'P':12,'Q':13,'R':14,'S':15,'T':16,
        'V':17,'W':18,'Y':19} # this mapping is a consequence of ivan's gamma definitions

    nwell = 1
    gamma_ijm = np.zeros((nwell, 20, 20))
    for i in range(20):
        for j in range(i,20):
            gamma_ijm[0,i,j] = pairwise_gamma[i,j]
            gamma_ijm[0,j,i] = pairwise_gamma[i,j]

    # create Force
    contact = CustomGBForce()
    contact.addTabulatedFunction("gamma_ijm", Discrete3DFunction(nwell, 20, 20, gamma_ijm.T.flatten()))
    contact.addTabulatedFunction("burial_gamma", Discrete1DFunction(burial_gamma))
    contact.addPerParticleParameter("resName")
    contact.addPerParticleParameter("resId")
    contact.addPerParticleParameter("isSite")
    contact.addPerParticleParameter("chainId")


    # quantities used to calculate both rho and 
    # pairwise interaction (directly, and indirectly through its reliance on rho)
    switching_function = f'(1/(1+exp(-25*({r_max}-r))))' 
    # Logic that evaluates to 1 if two residues should be included in the
    # rho/omega calculation and the contact energy calculation,
    # otherwise 0. This logic enforces a minimum sequence separation of 3
    # for two residues within the same chain (but not minimum for interchain pairs)
    seqsep = 'max(step(abs(resId1-resId2)-3),step(abs(chainId2-chainId1)-1))' 
    
    # tell the Force how to calculate the density/number of neighbors,
    # called "Omega^i" in the coluzza literature and "rho" or "rho_i" in the wolynes literature
    contact.addComputedValue("rho", 
        f"isSite1*isSite2*{seqsep}*{switching_function}", 
        CustomGBForce.ParticlePair)

    # we must add all Particles in the System to the Force, but
    # we activate isSite for one particle per residue 
    # replace cb with ca for GLY
    cb_fixed = [x if x > 0 else y for x,y in zip(cat.cb,cat.ca)]
    none_cb_fixed = [i for i in range(cat.natoms) if i not in cb_fixed]
    assert len(cb_fixed) == cat.nres, f"Number of atoms in cb_fixed (non-GLY CB and GLY CA atoms), {len(cb_fixed)}, does not match number of residues {cat.nres}."
    for i in range(cat.natoms):
        resName = letter_to_index[cat.seq[cat.resi[i]]]
        contact.addParticle([
            resName, # resName
            cat.resi[i],    # resId
            int(i in cb_fixed),  # isSite        
            find_chain_index(cat.resi[i], cat.chain_starts, cat.chain_ends), # chainId
            ]) 


    
    # Add pairwise term
    #     switching function goes from 0 to 1 and k_contact > 0,
    #     but we have the leading coefficient of -1, 
    #     so positive gammas make the overall energy negative
    energy_string = f"-isSite1*isSite2*{k_contact}*{seqsep}*gamma_ijm(1,resName1,resName2)*{switching_function}"
    contact.addEnergyTerm(energy_string, CustomGBForce.ParticlePair)

    # add burial term
    #     OMEGA-rho > 0 --> exposed
    #     OMEGA-rho < 0 --> buried
    #     so exposed-favoring residues should have negative gamma to penalize OMEGA-rho < 0
    #     while buried-favoring residues should have positive gamma to penalize OMEGA-rho > 0
    contact.addEnergyTerm(
        f"isSite*{k_burial}*max(0,burial_gamma(resName)*({OMEGA}-rho))", # this is what's written in coluzza 2011, but the code might be different
         CustomGBForce.SingleParticle)


    # boilerplate
    contact.setCutoffDistance(1.5) # needs to be bigger than the 12 angstroms
    if cat.periodic_box:
        contact.setNonbondedMethod(contact.CutoffPeriodic)
    else:
        contact.setNonbondedMethod(contact.CutoffNonPeriodic)
    contact.setForceGroup(forceGroup)
    
    return contact


def caterpillar_hb_term(cat, e_H, nu, sigma, forceGroup):
    # IMPLEMENTATION NOTE:
    # Currently, it is probably necessary for all instances of nonbonded forces
    # (NonbondedForce, CustomNonbondedForce, CustomGBForce) to share the same exclusion
    # list (see https://github.com/openmm/openmm/issues/5165). Even if the code compiles
    # on a particular Platform, it is possible that the results would be unreliable.
    # For this reason, it is risky to add Exclusions to nonbonded forces such as
    # caterpillar_contact_term, and we have written these terms to avoid using exclusions.
    # However, CustomHbondForce, used here, is not implemented as a nonbonded force
    # and doesn't use a neighbor list (github.com/openmm/openmm/pull/4060).
    # Unfortunately, the number of exclusions that may be added per donor/acceptor
    # is limited on some Platforms. 
    # So we will build the exclusions into the potential energy expression.
    #
    # create Force
    hb = CustomHbondForce(
        f"-{e_H}*max(step(abs(resIdD-resIdA)-4),step(abs(chainIdD-chainIdA)-1))*\
            ((cos(theta1)*cos(theta2))^{nu})*(5*(div^12)-6*(div^10))\
            ;div={sigma}/distance(a2,d2);theta1=angle(a1,a2,d2);theta2=angle(d1,d2,a2)")
    hb.addPerDonorParameter('resIdD')
    hb.addPerAcceptorParameter('resIdA')
    hb.addPerDonorParameter('chainIdD')
    hb.addPerAcceptorParameter('chainIdA')
    # add "donors" and "acceptors" that may interact with each other
    for resindex, oneletter in enumerate(cat.seq):
        assert cat.o[resindex] != -1 # every residue should have an O
        if resindex in cat.chain_starts: # residue shouldn't have an N or H
            assert cat.n[resindex] == -1
            assert cat.h[resindex] == -1
            assert cat.c[resindex] != -1
            hb.addAcceptor(cat.c[resindex], cat.o[resindex], -1, # -1 is a placeholder for the optional 3rd particle
                [resindex, find_chain_index(resindex, cat.chain_starts, cat.chain_ends)]) 
        elif resindex in cat.chain_ends: # residue shouldn't have a C
            assert cat.c[resindex] == -1
            assert cat.n[resindex] != -1
            if oneletter == "P":
                assert cat.h[resindex] == -1
            else:
                assert cat.h[resindex] != -1
                hb.addDonor(cat.n[resindex], cat.h[resindex], -1,
                    [resindex, find_chain_index(resindex, cat.chain_starts, cat.chain_ends)])
        else: # n and c should always be present, also h if not proline
            assert cat.c[resindex] != -1 
            assert cat.n[resindex] != -1
            hb.addAcceptor(cat.c[resindex], cat.o[resindex], -1,
                           [resindex, find_chain_index(resindex, cat.chain_starts, cat.chain_ends)])
            if oneletter == "P":
                assert cat.h[resindex] == -1
            else:
                assert cat.h[resindex] != -1
                hb.addDonor(cat.n[resindex], cat.h[resindex], -1,
                            [resindex, find_chain_index(resindex, cat.chain_starts, cat.chain_ends)]) 
    # boilerplate
    hb.setCutoffDistance(1.0) # this potential decays pretty quickly; cutoff considers d1-a1 distance
    if cat.periodic_box:
        hb.setNonbondedMethod(hb.CutoffPeriodic)
    else:
        hb.setNonbondedMethod(hb.CutoffNonPeriodic)
    hb.setForceGroup(forceGroup)                
    return hb
    
