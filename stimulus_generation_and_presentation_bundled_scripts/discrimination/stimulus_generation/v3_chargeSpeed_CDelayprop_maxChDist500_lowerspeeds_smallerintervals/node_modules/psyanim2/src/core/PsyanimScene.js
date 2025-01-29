import Phaser from 'phaser';

import PsyanimScreenBoundary from './utils/PsyanimScreenBoundary.js';
import PsyanimEntity from './PsyanimEntity.js';

import {
    PsyanimDebug
} from 'psyanim-utils';

/**
 *  In Psyanim 2.0, everything that is seen in the world lives in a `PsyanimScene`, which is an abstraction for the 2D world we are simulating.
 *
 *  A `PsyanimScene` IS a `Phaser.Scene`, and inherits all of its properties and methods.
 * 
 *  The `PsyanimScene` acts as a container for PsyanimEntity objects.
 */
export default class PsyanimScene extends Phaser.Scene {

    /**
     *  If set to `true`, all textures in texture manager will be deleted when this scene shuts down.
     * 
     * @type {boolean}
     */
    deleteTexturesOnShutdown;

    /**
     *  A reference to the `PsyanimScreenBoundary` used by this scene.
     * 
     * @type {PsyanimScreenBoundary}
     */
    screenBoundary;

    constructor(key) {

        super(key);

        this._afterCreateCalled = false;
    }

    get timeSinceLastInit() {

        return this._timeSinceLastInit;
    }

    /**
     * This method adds a new `PsyanimEntity` to the scene.
     * 
     * @param {string} name - entity name, must be unique per scene!
     * @param {Number} [x] - x-coordinate of entity's initial position in the world
     * @param {Number} [y] - y-coordinate of entity's initial position in the world
     * @param {Object} [shapeParams] - object defines entity shape & color
     * @param {Object} [matterOptions] - object defines physics properties of entity for matter-js
     * @returns {PsyanimEntity} - reference to newly created entity
     */
    addEntity(name, x = 0, y = 0, shapeParams = { isEmpty: true }, matterOptions = {}) {

        if (this._entities.some(e => e.name == name))
        {
            PsyanimDebug.error("Scene '" + this.scene.key + "' already has an entity named '" + name + "'!");
            return null;
        }

        let entity = new PsyanimEntity(this, name, x, y, shapeParams, matterOptions);

        this._entities.push(entity);

        if (this._afterCreateCalled)
        {
            entity.afterCreate();
        }

        return entity;
    }

    /**
     * This method adds a new 'PsyanimEntity' to the scene, using the `PsyanimEntityPrefab` to construct it.
     * 
     * @param {PsyanimEntityPrefab} prefab - instance of prefab that entity will be constructed from
     * @param {string} instanceName - must be unique per scene!
     * @param {Number} [x] - x-coordinate of entity's initial position in scene
     * @param {Number} [y] - y-coordinate of entity's initial position in scene
     * @returns {PsyanimEntity} - reference to newly created entity
     */
    instantiatePrefab(prefab, instanceName, x = 0, y = 0) {

        let entity = this.addEntity(
            instanceName, x, y,
            prefab.shapeParams, 
            prefab.matterOptions);

        prefab.create(entity);

        return entity;
    }

    /**
     * Get a list of all entity names in this scene.
     * @returns {string[]}
     */
    getAllEntityNames() {

        return this._entities.map(e => e.name);
    }

    get entities() {

        return this._entities.slice();
    }

    /**
     * Get a reference to an entity in this scene by name.
     * @param {string} name 
     * @returns {PsyanimEntity}
     */
    getEntityByName(name) {

        return this._entities.find(e => e.name == name);
    }

    /**
     * Removes entity from this scene by name and destroys it.
     * @param {string} name 
     */
    destroyEntityByName(name) {

        let entity = this._entities.find(e => e.name == name);

        if (entity) {

            this._entities = this._entities.filter(e => e.name != entity.name);
            entity.destroy();
        }
        else
        {
            console.warn("WARNING: failed to find entity named '" + name + "'!");
        }
    }

    /**
     * Searches all entities in this scene for `componentType` and returns a reference to the first component found.
     * @param {Type} componentType 
     * @returns {PsyanimComponent} the first component found of type 'componentType'
     */
    getComponentByType(componentType) {

        let entity = this._entities.find(e => e.getComponent(componentType) != null);

        if (entity != null)
        {
            return entity.getComponent(componentType);
        }

        return null;
    }

    /**
     * Searches all entities in this scene for `componentType` and returns a list of all components found.
     * @param {Type} componentType 
     * @returns {PsyanimComponent[]}
     */
    getComponentsByType(componentType) {

        return this._entities
            .filter(e => e.getComponent(componentType) != null)
            .map(e => e.getComponent(componentType));
    }

    /**
     *  The base `init()` method for all classes inheriting from `PsyanimScene`.
     */
    init() {

        this._timeSinceLastInit = this.time.now;

        this._entities = [];

        this.registry.set('psyanim_currentScene', this);

        this.deleteTexturesOnShutdown = true;

        // setup hook into the Phaser.Scene 'create' and 'shutdown' events only *once*
        if (!this._boundAfterCreate)
        {
            this._boundAfterCreate = this.afterCreate.bind(this);

            this.events.on('create', this._boundAfterCreate);    
        }

        if (!this._boundBeforeShutdown)
        {
            this._boundBeforeShutdown = this.beforeShutdown.bind(this);

            this.events.on('shutdown', this._boundBeforeShutdown);
        }
    }

    /**
     *  The base `preload()` method for all classes inheriting from `PsyanimScene`.
     */
    preload() {

    }

    /**
     *  The base `create()` method for all classes inheriting from 'PsyanimScene'.
     * 
     *  `create()` is called once everytime a scene is loaded.
     */
    create() {

        // setup wrapping with screen boundary
        this._canvasWidth = this.game.config.width;
        this._canvasHeight = this.game.config.height;
        let centerX = this._canvasWidth / 2;
        let centerY = this._canvasHeight / 2;

        this.screenBoundary = new PsyanimScreenBoundary(this, centerX, centerY, 
            this._canvasWidth, this._canvasHeight);
    }

    /**
     *  The base `afterCreate()` method for all classes inheriting from `PsyanimScene.`
     * 
     *  `afterCreate()` is called once after every call to `create()`.
     */
    afterCreate() {
 
        this._entities.forEach(e => e.afterCreate());

        this._afterCreateCalled = true;
    }

    /**
     *  Calling this method will delete all textures in a scene.
     * 
     *  WARNING: make sure you have deleted any entities that might reference these textures before deleting them!
     */
    deleteAllTextures() {

        let textureKeys = this.textures.getTextureKeys();

        textureKeys.forEach(key => {

            // TODO: verify if the textures are still being referenced by entities in the scene
            let isTextureReferenced = false;

            if (!isTextureReferenced)
            {
                this.textures.remove(key);
            }
            else
            {
                PsyanimDebug.warn('Failed to delete texture: ', key, '... it is still referenced.');
            }
        });
    }

    /**
     *  The base `beforeShutdown()` method for all classes inheriting from `PsyanimScene.`
     * 
     *  `beforeShutdown()` is called once before this scene is shutdown (usually before loading another scene).
     */
    beforeShutdown() {

        if (this.deleteTexturesOnShutdown)
        {
            this.deleteAllTextures();
        }

        this._entities.forEach(e => e.beforeShutdown());
    }

    /**
     * This is the main real-time update() loop for the entire `PsyanimScene`.
     * 
     * All entities & their components are updated in this loop.
     * 
     * @param {Number} t - The current simulation time in seconds
     * @param {Number} dt - The delta time in ms since the last frame. This is a smoothed and capped value based on the FPS rate.
     */
    update(t, dt) {

        this._entities.forEach(e => e.update(t, dt));
    }
}