/**
 * This is the base class for all `prefabs` in psyanim-2.
 */
export default class PsyanimEntityPrefab {

    /**
     *  Parameters defining the shape of the entity.
     * 
     *  @return {Object}
     *  @property {boolean} isEmpty - If `true`, this entity will not have a visual representation or physics applied.
     *  @property {PsyanimConstants.SHAPE_TYPE} shapeType - can be a circle, rectangle, or triangle
     *  @property {number} radius - If `shapeType` is `PsyanimConstants.SHAPE_TYPE.CIRCLE`, `radius` parameter controls its size.
     *  @property {number} width - If `shapeType` is `PsyanimConstants.SHAPE_TYPE.RECTANGLE`, `width` and `height` parameters control its size.
     *  @property {number} height - If `shapeType` is `PsyanimConstants.SHAPE_TYPE.RECTANGLE`, `width` and `height` parameters control its size.
     *  @property {number} base - If `shapeType` is `PsyanimConstants.SHAPE_TYPE.TRIANGLE`, `base` and `altitude` parameters control its size.
     *  @property {number} altitude - If `shapeType` is `PsyanimConstants.SHAPE_TYPE.TRIANGLE`, `base` and `altitude` parameters control its size.
     *  @property {number} color - integer representing RGB color
     *  @property {boolean} visible - `true` if entity is visible
     */
    shapeParams;

    /**
     *  Parameters defining the physics settings of the entity.
     * 
     *  @return {Object}
     *  @property {boolean} isSensor - If true, entity will be able to emit collision events, but will not be receive physics updates
     *  @property {boolean} isSleeping - If true, entity will not have physics updates or receive collision events
     *  @property {Object} collisionFilter - matter-js collision filter
     */
    matterOptions;

    /**
     * All prefabs must be constructed before they can be instantiated in the PsyanimScene.
     * 
     * A single prefab object may be instantiated in the scene multiple times.
     * 
     * During construction, `shapeParams` and `matterOptions` for the entity may optionally be supplied.
     * 
     * @param {Object} shapeParams 
     * @param {Object} matterOptions 
     */
    constructor(shapeParams = { isEmpty: true }, matterOptions = {}) {

        this.shapeParams = shapeParams;
        this.matterOptions = matterOptions;
    }

    /**
     * This `create()` method is intended to be overriden by the child class.
     * 
     * @param {PsyanimEntity} entity - the entity this prefab will work on during instantiation
     * @returns {PsyanimEntity} - the entity this prefab used during instantiation
     */
    create(entity) {

        return entity;
    }
}