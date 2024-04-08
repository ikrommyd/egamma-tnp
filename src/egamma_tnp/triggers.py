from egamma_tnp import TagNProbeFromNanoAOD, TagNProbeFromNTuples


class ElePt_WPTight_Gsf:
    """Tag and Probe efficiency for HLT_ElePt_WPTight_Gsf trigger.

    Parameters
    ----------
        fileset : dict
            The fileset to calculate the trigger efficiencies for.
        trigger_pt : int or float
            The Pt threshold of the trigger.
        from_ntuples : bool, optional
            Whether the fileset is E/Gamma NTuples or NanoAOD. The default is False.
        avoid_ecal_transition_tags : bool, optional
            Whether to avoid the ECAL transition region for the tags with an eta cut. The default is True.
        avoid_ecal_transition_probes : bool, optional
            Whether to avoid the ECAL transition region for the probes with an eta cut. The default is False.
        goldenjson : str, optional
            The golden json to use for luminosity masking. The default is None.
        extra_filter : Callable, optional
            An extra function to filter the events. The default is None.
            Must take in a coffea NanoEventsArray and return a filtered NanoEventsArray of the events you want to keep.
        extra_filter_args : dict, optional
            Extra arguments to pass to extra_filter. The default is {}.
    """

    def __new__(
        cls,
        fileset,
        trigger_pt,
        *,
        from_ntuples=False,
        avoid_ecal_transition_tags=True,
        avoid_ecal_transition_probes=False,
        goldenjson=None,
        extra_filter=None,
        extra_filter_args=None,
    ):
        if from_ntuples:
            # Create an instance of TagNProbeFromNTuples
            instance = TagNProbeFromNTuples.__new__(TagNProbeFromNTuples)
            TagNProbeFromNTuples.__init__(
                instance,
                fileset=fileset,
                filter=f"passHltEle{trigger_pt}WPTightGsf",
                tags_pt_cut=30,
                probes_pt_cut=trigger_pt - 3,
                tags_abseta_cut=2.5,
                cutbased_id="passingCutBasedTight122XV1",
                goldenjson=goldenjson,
                extra_filter=extra_filter,
                extra_filter_args=extra_filter_args,
                use_sc_eta=True,
                use_sc_phi=False,
            )
        else:
            # Create an instance of TagNProbeFromNanoAOD
            instance = TagNProbeFromNanoAOD.__new__(TagNProbeFromNanoAOD)
            TagNProbeFromNanoAOD.__init__(
                instance,
                fileset=fileset,
                filter=f"passHltEle{trigger_pt}WPTightGsf",
                for_trigger=True,
                trigger_pt=trigger_pt,
                tags_pt_cut=30,
                probes_pt_cut=trigger_pt - 3,
                tags_abseta_cut=2.5,
                filterbit=1,
                cutbased_id=4,
                goldenjson=goldenjson,
                extra_filter=extra_filter,
                extra_filter_args=extra_filter_args,
                use_sc_eta=True,
                use_sc_phi=False,
                avoid_ecal_transition_tags=avoid_ecal_transition_tags,
                avoid_ecal_transition_probes=avoid_ecal_transition_probes,
                hlt_filter=f"Ele{trigger_pt}_WPTight_Gsf",
            )

        return instance


class ElePt_CaloIdVT_GsfTrkIdT:
    """Tag and Probe efficiency for HLT_ElePt_CaloIdVT_GsfTrkIdT trigger.

    Parameters
    ----------
        fileset : dict
            The fileset to calculate the trigger efficiencies for.
        trigger_pt : int or float
            The Pt threshold of the trigger.
        from_ntuples : bool, optional
            Whether the fileset is E/Gamma NTuples or NanoAOD. The default is False.
        avoid_ecal_transition_tags : bool, optional
            Whether to avoid the ECAL transition region for the tags with an eta cut. The default is True.
        avoid_ecal_transition_probes : bool, optional
            Whether to avoid the ECAL transition region for the probes with an eta cut. The default is False.
        goldenjson : str, optional
            The golden json to use for luminosity masking. The default is None.
        extra_filter : Callable, optional
            An extra function to filter the events. The default is None.
            Must take in a coffea NanoEventsArray and return a filtered NanoEventsArray of the events you want to keep.
        extra_filter_args : dict, optional
            Extra arguments to pass to extra_filter. The default is {}.
    """

    def __new__(
        cls,
        fileset,
        trigger_pt,
        *,
        from_ntuples=False,
        avoid_ecal_transition_tags=True,
        avoid_ecal_transition_probes=False,
        goldenjson=None,
        extra_filter=None,
        extra_filter_args=None,
    ):
        if from_ntuples:
            # Create an instance of TagNProbeFromNTuples
            instance = TagNProbeFromNTuples.__new__(TagNProbeFromNTuples)
            TagNProbeFromNTuples.__init__(
                instance,
                fileset=fileset,
                filter=f"passHltEle{trigger_pt}CaloIdVTGsfTrkIdTGsf",
                tags_pt_cut=30,
                probes_pt_cut=trigger_pt - 3,
                tags_abseta_cut=2.5,
                cutbased_id="passingCutBasedTight122XV1",
                goldenjson=goldenjson,
                extra_filter=extra_filter,
                extra_filter_args=extra_filter_args,
                use_sc_eta=True,
                use_sc_phi=False,
            )
        else:
            # Create an instance of TagNProbeFromNanoAOD
            instance = TagNProbeFromNanoAOD.__new__(TagNProbeFromNanoAOD)
            TagNProbeFromNanoAOD.__init__(
                instance,
                fileset=fileset,
                filter=f"passHltEle{trigger_pt}CaloIdVTGsfTrkIdTGsf",
                for_trigger=True,
                trigger_pt=trigger_pt,
                tags_pt_cut=30,
                probes_pt_cut=trigger_pt - 3,
                tags_abseta_cut=2.5,
                filterbit=11,
                cutbased_id=4,
                goldenjson=goldenjson,
                extra_filter=extra_filter,
                extra_filter_args=extra_filter_args,
                use_sc_eta=True,
                use_sc_phi=False,
                avoid_ecal_transition_tags=avoid_ecal_transition_tags,
                avoid_ecal_transition_probes=avoid_ecal_transition_probes,
                hlt_filter=f"Ele{trigger_pt}_CaloIdVT_GsfTrkIdT",
            )

        return instance
