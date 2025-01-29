import Phaser from 'phaser';

import PsyanimConstants from './PsyanimConstants.js';
import PsyanimGeomUtils from './utils/PsyanimGeomUtils.js';

import PsyanimApp from './PsyanimApp.js';

import {
    PsyanimDebug
} from 'psyanim-utils';

/**
 *  Any object that exists in a scene, regardless of its visual representation, is called a `PsyanimEntity`.
 *
 *  More technically, it's an abstraction for anything that exists in a `Psyanim Scene` with a particular location, rotation, and (optionally) a visual representation which can have physics applied to it.
 *
 *  Entities may or may not have a visual representation in the scene.
 *
 *  Moreover, entities alone do not have any logic or behaviors. While entities have no user-defined state, they do have a position, orientation and velocity in the world, and can have forces / accelerations applied to them.
 *
 *  Any object in your simulation that needs a position or orientation, a visual representation in the scene, or needs to have physics applied to it, should be added to the scene as an entity.
 * 
 *  A `PsyanimEntity` acts a container for `PsyanimComponents`.
 * 
 *  All user-defined state and behaviors are encapsulated in PsyanimComponents, which are reusable scripts that can be attached to a PsyanimEntity and offer hooks into the real-time update loop of each scene.
 *
 *  -----------------------------------------------------------------------------------------------------------------
 * 
 *  Some helpful tips:
 *      - use 'this.visible' to toggle sprite visibility
 *      - use 'this.body.isSleeping' to toggle whether or not this object receives physics updates
 *      - use 'this.body.isSensor' to toggle whether this body collides with other bodies or not
 * 
 *  From: https://github.com/liabru/matter-js/issues/179
 * 
 *  "Internally the engine uses MKS (meters, kilograms, and seconds) units 
 *  and radians for angles.
 * 
 *  If you use the built in renderer and and built in runner, with default 
 *  settings this translates to:
 *  
 *      1 position = 1 px
 *      1 speed = 1 px per step
 *      1 step = 16.666ms"
 */
export default class PsyanimEntity extends Phaser.Physics.Matter.Sprite {

    /**
     * Constructor for PsyanimEntity objects.
     * 
     * To create entities in your app, use the PsyanimScene::addEntity() method.
     * 
     * This is usually only called by the PsyanimScene.
     * 
     * @param {PsyanimScene} scene - scene this entity belongs to.
     * @param {string} name - entity name, must be unique per scene!
     * @param {Number} [x] - x-coordinate of entity's initial position in the world
     * @param {Number} [y] - y-coordinate of entity's initial position in the world
     * @param {Object} [shapeParams] - object defines entity shape & color
     * @param {Object} [matterOptions] - object defines physics properties of entity for matter-js
     */
    constructor(scene, name, x = 0, y = 0, shapeParams = { isEmpty: true }, matterOptions = {}) {

        // before we call super(), let's generate the necessary textures and matter config
        let textureKey = null;

        if (shapeParams.textureKey) 
        {
            textureKey = shapeParams.textureKey;
        }
        else
        {
            textureKey = PsyanimApp.Instance.currentSceneKey + "_" + name;
        }

        const defaultShapeParams = PsyanimConstants.DEFAULT_ENTITY_SHAPE_PARAMS;

        let shapeType = Object.hasOwn(shapeParams, 'shapeType') ? shapeParams.shapeType : defaultShapeParams.shapeType;
        let color = Object.hasOwn(shapeParams, 'color') ? shapeParams.color : defaultShapeParams.color;

        let geomParams = null;
        let matterConfig = { type: 'circle', radius: 1 };

        let isEmpty = Object.hasOwn(shapeParams, 'isEmpty') ? shapeParams.isEmpty : false;

        if (isEmpty)
        {
            textureKey = '';
        }
        else
        {
            let generateNewTexture = false;

            if ( !scene.textures.exists(textureKey) )
            {
                generateNewTexture = true;
            }    

            switch(shapeType)
            {
                case PsyanimConstants.SHAPE_TYPE.CIRCLE:

                    let radius = (shapeParams.radius) ? shapeParams.radius : defaultShapeParams.radius;

                    let circleGeomParams = {
                        radius: radius
                    };

                    geomParams = circleGeomParams;

                    if (generateNewTexture)
                    {
                        PsyanimGeomUtils.generateCircleTexture(scene, textureKey, circleGeomParams, color);
                    }

                    matterConfig.type = 'circle';
                    matterConfig.radius = circleGeomParams.radius;

                    break;

                case PsyanimConstants.SHAPE_TYPE.TRIANGLE:

                    let base = (shapeParams.base) ? shapeParams.base : defaultShapeParams.base;
                    let altitude = (shapeParams.altitude) ? shapeParams.altitude : defaultShapeParams.altitude;
        
                    let triangleGeomParams = {
                        base: base, altitude: altitude
                    };

                    geomParams = triangleGeomParams;

                    if (generateNewTexture)
                    {
                        PsyanimGeomUtils.generateTriangleTexture(scene, textureKey, triangleGeomParams, color);
                    }

                    matterConfig.type = 'fromVertices';
                    matterConfig.verts = PsyanimGeomUtils.computeTriangleVertices(base, altitude);
        
                    break;

                case PsyanimConstants.SHAPE_TYPE.RECTANGLE:

                    let width = (shapeParams.width) ? shapeParams.width : defaultShapeParams.width;
                    let height = (shapeParams.height) ? shapeParams.height : defaultShapeParams.height;

                    let rectangleGeomParams = {
                        width: width, height: height
                    };

                    geomParams = rectangleGeomParams;

                    if (generateNewTexture)
                    {
                        PsyanimGeomUtils.generateRectangleTexture(scene, textureKey, rectangleGeomParams, color);
                    }

                    matterConfig.type = 'rectangle';
                    matterConfig.width = width;
                    matterConfig.height = height;

                    break;
            }
        }

        // initialize Phaser.Physics.Matter.Sprite
        super(scene.matter.world, x, y, textureKey);

        // save off this entity's properties
        this._shapeParams = shapeParams;
        this._matterOptions = matterOptions;
        this._initialPosition = { x: x, y: y };

        this._color = color;
        this._geomParams = geomParams;
        this._shapeType = shapeType;
        
        this.name = name;

        // setup new physics body
        matterOptions.name = Object.hasOwn(matterOptions, 'name') ? matterOptions.name : name;
        matterOptions.isSensor = Object.hasOwn(matterOptions, 'isSensor') ? matterOptions.isSensor : false;
        matterOptions.isSleeping = Object.hasOwn(matterOptions, 'isSleeping') ? matterOptions.isSleeping : false;
        matterOptions.collisionFilter = Object.hasOwn(matterOptions, 'collisionFilter') 
            ? matterOptions.collisionFilter : PsyanimConstants.DEFAULT_SPRITE_COLLISION_FILTER;
        
        this.setBody(matterConfig, matterOptions);

        // there's a bug in phaser where 'label' doesn't get set for non-primitive bodies
        this.body.label = this.name;

        // setup visibility and isSensor / isSleeping depending on whether this object is empty
        this.body.isSleeping = matterOptions.isSleeping;

        if (isEmpty)
        {
            this.visible = false;

            this.body.isSensor = true;
        }
        else
        {
            this.visible = Object.hasOwn(shapeParams, 'visible') ? shapeParams.visible : defaultShapeParams.visible;

            this.body.isSensor = matterOptions.isSensor;
        }

        if (Object.hasOwn(shapeParams, 'depth'))
        {
            this.depth = shapeParams.depth;
        }

        let mass = 100;

        this.body.mass = mass;
        this.body.inverseMass = 1/mass;

        this.body.inertia = Infinity;
        this.body.inverseInertia = 0;

        // we add the entity to the scene in constructor to guarantee all entities always live in a scene
        scene.add.existing(this);

        // setup private fields
        this._components = [];
    }

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
    get shapeParams() {

        return this._shapeParams;
    }

    /**
     *  Parameters defining the physics settings of the entity.
     * 
     *  @return {Object}
     *  @property {boolean} isSensor - If true, entity will be able to emit collision events, but will not be receive physics updates
     *  @property {boolean} isSleeping - If true, entity will not have physics updates or receive collision events
     *  @property {Object} collisionFilter - matter-js collision filter
     */
    get matterOptions() {

        return this._matterOptions;
    }

    /**
     *  Entity shape type (defined in PsyanimConstants)
     * 
     *  @return {string}
     */
    get shapeType() {
        return this._shapeType;
    }

    /**
     *  Entity color
     * 
     *  @return {number}
     */
    get color() {
        return this._color;
    }

    /**
     *  Set's entity's color
     * 
     *  @param {number} - RGB color value
     */
    set color(value) {

        this.setTintFill(value);
    }

    /**
     *  Entity's initial position, supplied on construction
     * 
     *  @return {Object}
     *  @property {number} - x
     *  @property {number} - y
     */
    get initialPosition() {

        return this._initialPosition;
    }

    /**
     * Enabled friction on entity in physics simulation
     * 
     * @param {number} friction - 
     * @param {number} frictionAir - 
     * @param {number} frictionStatic - 
     */
    enableFriction(friction = 0.1, frictionAir = 0.01, frictionStatic = 0.5) {

        this.body.friction = friction;
        this.body.frictionAir = frictionAir;
        this.body.frictionStatic = frictionStatic;
    }

    /**
     * Disables friction on entity in physics simulation
     */
    disableFriction() {

        this.body.friction = 0;
        this.body.frictionAir = 0;
        this.body.frictionStatic = 0;
    }

    /**
     * Attaches a newly created instance of `componentType` to this entity.
     * 
     * @param {Type} componentType - the type of the component that should be added to this entity
     * @returns {PsyanimComponent} - reference to newly added PsyanimComponent
     */
    addComponent(componentType) {

        if (this.hasComponent(componentType))
        {
            PsyanimDebug.error("Attempting to add the same component type, '" + componentType.constructor.name +
                "' more than once to entity '" + this.name + "'!");

            return null;
        }

        let newComponent = new componentType(this);

        this._components.push(newComponent);

        return newComponent;
    }

    /**
     * @param {Type} componentType 
     * @returns {boolean} - `true` if this entity has at least 1 component attached of type `componentType`
     */
    hasComponent(componentType) {

        return this.getComponent(componentType) != null;
    }

    /**
     * Get an attached component instance on this entity by type.
     * 
     * @param {Type} componentType 
     * @returns {PsyanimComponent}
     */
    getComponent(componentType) {

        let components = this._components.filter(c => c instanceof componentType);

        if (components.length == 0)
        {
            return null;
        }

        return components[0];
    }

    /**
     * Get an attached component instance on this entity by index.
     * 
     * Component indices are based on the order they were added using `AddComponent`
     * 
     * @param {number} index 
     * @returns {PsyanimComponent} - reference to attached component instance at `index`
     */
    getComponentByIndex(index) {

        return this._components[index];
    }

    /**
     * Gets a shallow copy of PsyanimComponents attached to this entity
     * @returns {PsyanimComponent[]}
     */
    getComponents() {

        return this._components.slice();
    }

    /**
     * Enables / disables the physics simulation for this entity
     * 
     * @param {boolean} enabled 
     */
    setPhysicsEnabled(enabled) {

        this.body.isSensor = !enabled;
        this.body.isSleeping = !enabled;
    }

    /**
     * 
     * @returns An array of distances for each screen boundary as [left, right, top, bottom]
     */
    computeDistancesToScreenBoundaries() {

        return [
            this.position.x,
            this.scene._canvasWidth - this.position.x,
            this.position.y,
            this.scene._canvasHeight - this.position.y
        ];
    }
    
    /**
     * Destroys this entity, cleaning up all of its resources.
     * 
     * NOTE: When overriding this in a derived class, the base class destroy() should be called last.
     */
    destroy() {

        this._components.forEach(c => {
            c.destroy();
        });
        
        this._components = [];

        super.destroy();
    }

    _removeComponent(component) {

        this._components = this._components.filter(c => c != component);
    }

    /**
     * This entity's position in the scene.
     * 
     * @returns {Phaser.Math.Vector2}
     * @property {number} x
     * @property {number} y
     */
    get position() {

        return new Phaser.Math.Vector2(this.x, this.y);
    }

    /**
     *  Sets this entity's position in the scene.
     * 
     *  @param {Object} - vector of the form: { x, y }
     */
    set position(value) {

        /**
         * The x-coordinate of this entity's position in the scene.
         * 
         * @type {number}
         */
        this.x = value.x;

        /**
         * The y-coordinate of this entity's position in the scene.
         * 
         * @type {number}
         */
        this.y = value.y;
    }

    /**
     * A unit vector oriented in the direction this entity is facing.
     * 
     * @return {Phaser.Math.Vector2}
     */
    get forward() {

        return new Phaser.Math.Vector2(1, 0).setAngle(this.angle * Math.PI / 180);
    }

    /**
     * A unit vector oriented to the right of this entity's forward direction.
     * 
     * @return {Phaser.Math.Vector2} 
     */
    get right() {

        return this.forward.rotate(90 * Math.PI / 180);
    }

    /**
     * The velocity vector of this entity.
     * 
     * @return {Phaser.Math.Vector2}
     */
    get velocity() {

        let currentVelocityXY = this.scene.matter.body.getVelocity(this.body);

        return new Phaser.Math.Vector2(currentVelocityXY.x, currentVelocityXY.y);
    }

    afterCreate() {

        this._components.forEach(c => {

            c.afterCreate();
        });
    }

    beforeShutdown() {

        this._components.forEach(c => {

            c.beforeShutdown();
        });
    }

    update(t, dt) {

        this._components.forEach(c => {

            if (c.enabled)
            {
                c.update(t, dt);
            }
        });
    }
}