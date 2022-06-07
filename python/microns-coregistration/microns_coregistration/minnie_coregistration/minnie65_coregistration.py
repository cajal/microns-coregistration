"""
DataJoint tables for coregistration of minnie65 electron microscopy and two-photon data.
"""

import datajoint_plus as djp

from microns_coregistration_api.schemas import minnie65_coregistration as m65crg

schema = m65crg.schema
config = m65crg.config

logger = djp.getLogger(__name__)

class Adjustment(m65crg.Adjustment):
    contents = [
        {'adjustment': 'resize', 'description': 'adjustment to source resolution (e.g. px / cm) but not source dimensionality (# pixels does not change)'},
        {'adjustment': 'resample', 'description': 'adjustment to source dimensionality (e.g. interpolation on a new grid)'},
        {'adjustment': 'crop', 'description': ''}, 
        {'adjustment': 'translation', 'description': 'constant offset added to source grid'},
    ]


class AdjustmentSet(m65crg.AdjustmentSet):
    
    class Member(m65crg.AdjustmentSet.Member): pass

    @classmethod
    def fill_set(cls, adjustments:list):
        kwargs = dict(ignore_extra_fields=True, skip_duplicates=True)
        adjustments = [adjustments] if isinstance(adjustments, str) else adjustments
        entry = [{'adjustment': a} for a in adjustments]
        constant_attrs = {'name': '_'.join(adjustments)}  if len(adjustments) > 1 else {'name': f'{adjustments[0]}_only'}
        cls.insert(entry, constant_attrs=constant_attrs, insert_to_parts=cls.Member, **kwargs)
    
    @classmethod
    def fill(cls):
        for a in Adjustment:
            cls.fill_set(a['adjustment'])
        
        sets = [
            ['crop', 'resample']
        ]

        for s in sets:
            cls.fill_set(s)


class TPStack(m65crg.TPStack): pass


class Coregistration(m65crg.Coregistration):
    
    class AIBS(m65crg.Coregistration.AIBS):

        @classmethod
        def fill(cls):
            m65 = djp.create_djp_module(schema_name='microns_minnie65_02')
            cls.insert(m65.Coregistration, insert_to_master=True)