if __name__ == '__main__':
    from microns_coregistration.minnie_coregistration import minnie65_calcium_sim as m65cs
    m65cs.VoxelizedSoma.Maker.populate(reserve_jobs=True, order='random', suppress_errors=True)

