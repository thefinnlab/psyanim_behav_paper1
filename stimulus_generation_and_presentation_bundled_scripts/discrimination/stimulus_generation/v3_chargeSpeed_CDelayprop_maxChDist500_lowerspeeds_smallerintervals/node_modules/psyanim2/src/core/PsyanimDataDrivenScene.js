import PsyanimScene from './PsyanimScene.js';

import PsyanimNavigationGrid from './utils/PsyanimNavigationGrid.js';
import PsyanimComponent from './PsyanimComponent.js';
import PsyanimEntityPrefab from './PsyanimEntityPrefab.js';

import {
    PsyanimDebug,
    PsyanimUtils
} from 'psyanim-utils';

/**
 *  Algorithm (order is important!):
 * 
 *      - Validate scene definition adheres to schema
 *      - Setup navgrid, if configured
 *      - Instantiate all non-prefab entities
 *      - Add all components to non-prefab entities
 *      - Configure all prefabs' value-type fields (non-entity or non-components fields)
 *      - Instantiate all prefab entities in the scene
 *      - Configure all entities' components
 * 
 */

export default class PsyanimDataDrivenScene extends PsyanimScene {

    constructor(key = null) {

        let sceneKey = 'PsyanimDataDrivenScene';

        if (key)
        {
            sceneKey = key;
        }

        super(sceneKey);
    }

    create() {

        super.create();

        this._sceneDefinition = this.registry.get('psyanim_currentSceneDefinition');

        if (Object.hasOwn(this._sceneDefinition, 'wrapScreenBoundary'))
        {
            this.screenBoundary.wrap = this._sceneDefinition.wrapScreenBoundary;
        }

        this._validateSceneDefinition();
        this._setupNavgrid();
        this._instantiateNonPrefabEntities();
        this._instantiatePrefabEntities();
        this._configureEntities();
    }

    _validateSceneDefinition() {

        // validate scene definition exists
        if (!this._sceneDefinition)
        {
            PsyanimDebug.error('No valid scene definition configured!');
        }

        // validate 'wrapScreenBoundary'
        if (Object.hasOwn(this._sceneDefinition, 'wrapScreenBoundary'))
        {
            if (typeof(this._sceneDefinition.wrapScreenBoundary) != 'boolean')
            {
                PsyanimDebug.error("'wrapScreenBoundary' must be a boolean!");
            }
        }

        // validate 'navgrid' field
        if (Object.hasOwn(this._sceneDefinition, 'navgrid'))
        {
            if (Object.hasOwn(this._sceneDefinition.navgrid, 'cellSize'))
            {
                if (typeof(this._sceneDefinition.navgrid.cellSize) !== 'number')
                {
                    PsyanimDebug.error("navgrid 'cellSize' must be a 'number'!");
                }
            }

            if (Object.hasOwn(this._sceneDefinition.navgrid, 'obstacles'))
            {
                if (!Array.isArray(this._sceneDefinition.navgrid.obstacles))
                {
                    PsyanimDebug.error("navgrid 'obstacles' field must be an array!");
                }

                let obstacleDefinitions = this._sceneDefinition.navgrid.obstacles;

                for (let i = 0; i < obstacleDefinitions.length; ++i)
                {
                    let obstacleDefinition = obstacleDefinitions[i];

                    if (!obstacleDefinition.position || !PsyanimUtils.isObject(obstacleDefinition.position) ||
                        !Object.hasOwn(obstacleDefinition.position, 'x') ||
                        !Object.hasOwn(obstacleDefinition.position, 'y'))
                    {
                        PsyanimDebug.error("Navgrid obstacle at index '" + i + "' must have a 'position' field of the form: " + 
                            "{ x: 0, y: 0 }");
                    }

                    if (Object.hasOwn(obstacleDefinition, 'shapeParams'))
                    {
                        if (!PsyanimUtils.isObject(obstacleDefinition.shapeParams))
                        {
                            PsyanimDebug.error("Navgrid obstacle at index '" + i + "' 'shapeParams' field must be an object!");
                        }
                    }
                }
            }
        }

        // validate entity configurations
        if (Object.hasOwn(this._sceneDefinition, 'entities'))
        {
            if (!Array.isArray(this._sceneDefinition.entities))
            {
                PsyanimDebug.error("'entities' must be an array!");
            }

            let entityDefinitions = this._sceneDefinition.entities;

            // check entity name exists and is unique
            let entityNames = entityDefinitions.map(e => e.name);

            if (entityNames.includes(undefined))
            {
                let index = entityNames.indexOf(undefined);

                PsyanimDebug.error("Entity at index '" + index + "' has no 'name' field!");
            }

            if (entityNames.length !== new Set(entityNames).size) // uniqueness check
            {
                PsyanimDebug.error("All entity names must be unique!");
            }

            // now do the detailed checks on each entity
            for (let i = 0; i < entityDefinitions.length; ++i)
            {
                let entityDefinition = entityDefinitions[i];

                if (typeof(entityDefinition.name) != 'string')
                {
                    PsyanimDebug.error("Entity index '" + i + "': 'name' must be of type 'string'!");
                }

                if (Object.hasOwn(entityDefinition, 'instances'))
                {
                    if (typeof(entityDefinition.instances) !== 'number')
                    {
                        PsyanimDebug.error("Entity at index = " + i + ": 'instances' field must be a number!");
                    }
                }

                if (Object.hasOwn(entityDefinition, 'initialPosition'))
                {
                    if (!PsyanimUtils.isObject(entityDefinition.initialPosition) && entityDefinition.initialPosition !== 'random')
                    {
                        PsyanimDebug.error("Entity index '" + i + "' has an invalid 'initialPosition'!");
                    }

                    if (entityDefinition.initialPosition !== 'random')
                    {
                        if (!Object.hasOwn(entityDefinition.initialPosition, 'x'))
                        {
                            PsyanimDebug.error("Entity index '" + i + "' has an invalid 'initialPosition' - no 'x' value!");
                        }
    
                        if (!Object.hasOwn(entityDefinition.initialPosition, 'y'))
                        {
                            PsyanimDebug.error("Entity index '" + i + "' has an invalid 'initialPosition' - no 'y' value!");
                        }    
                    }
                }

                if (Object.hasOwn(entityDefinition, 'initialAngle'))
                {
                    if (typeof(entityDefinition.initialAngle) != 'number' && entityDefinition.initialAngle !== 'random')
                    {
                        PsyanimDebug.error("Entity index '" + i + "' has an invalid 'initialAngle'!");
                    }
                }

                if (Object.hasOwn(entityDefinition, 'shapeParams'))
                {
                    if (!PsyanimUtils.isObject(entityDefinition.shapeParams))
                    {
                        PsyanimDebug.error("Entity index '" + i + "' has invalid 'shapeParams' parameter - shapeParams must be an object!");   
                    }
                }

                // validate entity prefabs
                if (Object.hasOwn(entityDefinition, 'prefab'))
                {
                    if (!PsyanimUtils.isObject(entityDefinition.prefab))
                    {
                        PsyanimDebug.error("Entity index '" + i + "' has invalid 'prefab' parameter - must be an object!");
                    }

                    if (!Object.hasOwn(entityDefinition.prefab, 'type'))
                    {
                        PsyanimDebug.error("Entity index '" + i + "' has a 'prefab' parameter that's missing a 'type' field!");
                    }

                    if (!(entityDefinition.prefab.type.prototype instanceof PsyanimEntityPrefab))
                    {
                        PsyanimDebug.error("Entity index '" + i + "' has a 'prefab' type that doesn't inherit from PsyanimEntityPrefab!");
                    }

                    if (Object.hasOwn(entityDefinition.prefab, 'params'))
                    {
                        if (!PsyanimUtils.isObject(entityDefinition.prefab.params))
                        {
                            PsyanimDebug.error("Entity index '" + i + "' has an invalid prefab 'params' field.");
                        }

                        let paramKeys = Object.keys(entityDefinition.prefab.params);

                        for (let j = 0; j < paramKeys.length; ++j)
                        {
                            let paramKey = paramKeys[j];
                            let paramValue = entityDefinition.prefab.params[paramKey];
    
                            if (PsyanimUtils.isObject(paramValue) && Object.hasOwn(paramValue, 'entityName'))
                            {
                                PsyanimDebug.error("Entity index '" + i + "', prefab param index = '" + j + 
                                    "': prefab parameters shouldn't reference entities!");

                                if (Object.hasOwn(paramValue, 'componentType'))
                                {
                                    PsyanimDebug.error("Entity index '" + i + "', param key '" + paramKey + 
                                        "': prefab parameters shouldn't reference other components!");
                                }
                            }
                        }                        
                    }
                }

                // validate components on entities
                if (Object.hasOwn(entityDefinition, 'components'))
                {
                    let componentDefinitions = entityDefinition.components;

                    if (!Array.isArray(componentDefinitions))
                    {
                        PsyanimDebug.error("Entity index '" + i +"': 'components' field must be an array!");
                    }

                    for (let j = 0; j < componentDefinitions.length; ++j)
                    {
                        let componentDefinition = componentDefinitions[j];

                        if (!PsyanimUtils.isObject(componentDefinition) || !Object.hasOwn(componentDefinition, 'type'))
                        {
                            PsyanimDebug.error("Entity index '" + i + "', component index '" + j + 
                                "': Component definition must be an object with a 'type' field!");
                        }

                        if (!(componentDefinition.type.prototype instanceof PsyanimComponent))
                        {
                            PsyanimDebug.error("Entity index '" + i + "', component index '" + j + 
                                "': Component definition 'type' must be a type that inherits from PsyanimComponent!");
                        }

                        if (Object.hasOwn(componentDefinition, 'params'))
                        {
                            if (!PsyanimUtils.isObject(componentDefinition.params))
                            {
                                PsyanimDebug.error("Entity index '" + i + "', component index '" + j + 
                                "': Component 'params' must be an object!");
                            }

                            let paramKeys = Object.keys(componentDefinition.params);

                            for (let k = 0; k < paramKeys.length; ++k)
                            {
                                let paramKey = paramKeys[k];
                                let paramValue = componentDefinition.params[paramKey];
    
                                if (PsyanimUtils.isObject(paramValue))
                                {
                                    if (Object.hasOwn(paramValue, 'entityName'))
                                    {
                                        if (!entityNames.includes(paramValue.entityName))
                                        {
                                            PsyanimDebug.error("Entity index '" + i + "', component index '" + j + 
                                                "', param key '" + paramKey + "': entity name doesn't exist in scene definition!");
                                        }
        
                                        if (Object.hasOwn(paramValue, 'componentType'))
                                        {
                                            let componentType = paramValue.componentType;
                                            let entityDefinition = entityDefinitions.find(e => e.name == paramValue.entityName);
        
                                            if (!Object.hasOwn(entityDefinition, 'components'))
                                            {
                                                PsyanimDebug.warn("Entity index '" + i + "', component index '" + j + 
                                                "', param key '" + paramKey + "': entity doesn't have any configured components!");
                                            }
        
                                            let componentDefinition = entityDefinition.components
                                                .find(c => c.type == paramValue.componentType);

                                            if (!componentDefinition)
                                            {
                                                PsyanimDebug.error("Entity index '" + i + "', component type '" + typeof(paramValue.componentType) + 
                                                    "', param key '" + paramKey + "': component index is greater than number of components on entity!");
                                            }
                                        }
                                    }
                                }
                            }                            
                        }
                    }
                }
            }
        }
        else
        {
            PsyanimDebug.warn('No entities added to the scene definition!');
        }
    }

    _getRandomizedInitialPosition() {

        let canvasWidth = this.game.config.width;
        let canvasHeight = this.game.config.height;

        let canvasCenter = {
            x: canvasWidth / 2,
            y: canvasHeight / 2
        };

        let maxDeltaX = (canvasWidth - 100) / 2;
        let maxDeltaY = (canvasHeight - 100) / 2;

        let deltaX = (Math.random() * 2 - 1) * maxDeltaX;
        let deltaY = (Math.random() * 2 - 1) * maxDeltaY;

        return {
            x: canvasCenter.x + deltaX,
            y: canvasCenter.y + deltaY
        };
    }

    _configureEntityComponents(entityDefinition, nameOverride = null) {

        let name = (nameOverride) ? nameOverride : entityDefinition.name;

        let entityReference = this.getEntityByName(name);

        let componentDefinitions = entityDefinition.components;

        if (componentDefinitions)
        {
            for (let j = 0; j < componentDefinitions.length; ++j)
            {
                let componentDefinition = componentDefinitions[j];

                let componentReference = entityReference.getComponent(componentDefinition.type);

                if (Object.hasOwn(componentDefinition, 'enabled'))
                {
                    componentReference.enabled = componentDefinition.enabled;
                }

                let componentParams = componentDefinition.params;

                if (componentParams)
                {
                    let paramNames = Object.keys(componentParams);

                    for (let k = 0; k < paramNames.length; ++k)
                    {
                        let paramName = paramNames[k];

                        let paramValue = componentParams[paramName];

                        if (PsyanimUtils.isObject(paramValue) && Object.hasOwn(paramValue, 'entityName'))
                        {
                            // param value an entity or component reference
                            let targetEntityName = paramValue.entityName;
                            
                            let targetEntity = this.getEntityByName(targetEntityName);

                            if (Object.hasOwn(paramValue, 'componentType'))
                            {
                                // target is a component reference
                                let componentType = paramValue.componentType;
                                let targetComponent = targetEntity.getComponent(componentType);

                                componentReference[paramName] = targetComponent;
                            }
                            else
                            {
                                // target is an entity reference
                                componentReference[paramName] = targetEntity;
                            }
                        }
                        else
                        {
                            componentReference[paramName] = paramValue;
                        }
                    }
                }
            }    
        }
    }

    _configureEntities() {

        let entityDefinitions = this._sceneDefinition.entities;

        if (!entityDefinitions || entityDefinitions.length == 0)
        {
            return;
        }

        /** finally, configure all the components */
        for (let i = 0; i < entityDefinitions.length; ++i)
        {
            let entityDefinition = entityDefinitions[i];

            let nInstances = Object.hasOwn(entityDefinition, 'instances') ? 
                entityDefinition.instances : 1;

            if (nInstances < 1)
            {
                PsyanimDebug.error("'instances' can not be less than 1!  entity name: " + entityDefinition.name);
            }

            if (nInstances == 1)
            {
                this._configureEntityComponents(entityDefinition);
            }
            else
            {
                for (let j = 0; j < nInstances; ++j)
                {
                    let instanceName = entityDefinition.name + '_' + (j + 1);

                    this._configureEntityComponents(entityDefinition, instanceName);
                }    
            }
        }
    }

    _instantiatePrefabEntity(entityDefinition, nameOverride = null) {

        let name = (nameOverride) ? nameOverride : entityDefinition.name;

        // compute initial position
        let initialPosition = null;
        let angle = 0;

        if (Object.hasOwn(entityDefinition, 'initialPosition'))
        {
            if (entityDefinition.initialPosition === 'random')
            {
                initialPosition = this._getRandomizedInitialPosition();
            }
            else
            {
                initialPosition = entityDefinition.initialPosition;
            }
        }
        else
        {
            initialPosition = { x: 0, y: 0 };
        }

        if (Object.hasOwn(entityDefinition, 'initialAngle'))
        {
            if (entityDefinition.initialAngle === 'random')
            {
                angle = Math.random() * 360.0;
            }
            else
            {
                angle = entityDefinition.initialAngle;
            }
        }

        let shapeParams = Object.hasOwn(entityDefinition, 'shapeParams') ? 
            entityDefinition.shapeParams : { isEmpty: true};

        let matterOptions = Object.hasOwn(entityDefinition, 'matterOptions') ?
            entityDefinition.matterOptions : {};

        let prefab = new entityDefinition.prefab.type(shapeParams, matterOptions);

        /** configure the prefab first */
        if (Object.hasOwn(entityDefinition.prefab, 'params'))
        {
            let prefabParamNames = Object.keys(entityDefinition.prefab.params);

            for (let j = 0; j < prefabParamNames.length; ++j)
            {
                let paramName = prefabParamNames[j];
    
                let paramValue = entityDefinition.prefab.params[paramName];
    
                if (paramValue === 'navgrid')
                {
                    prefab[paramName] = this._grid;
                }
                else if (PsyanimUtils.isObject(paramValue) && Object.hasOwn(paramValue, 'entityName'))
                {
                    PsyanimDebug.error('Prefab parameters should not reference other entities or components!');
                }
                else
                {
                    prefab[paramName] = paramValue;
                }
            }
        }

        /** instantiate the prefab after it's been properly configured */
        let newEntity = this.instantiatePrefab(prefab, name,
            initialPosition.x, initialPosition.y);

        newEntity.angle = angle;
    }

    _instantiatePrefabEntities() {

        let entityDefinitions = this._sceneDefinition.entities;

        if (!entityDefinitions || entityDefinitions.length == 0)
        {
            return;
        }

        let prefabEntityDefinitions = entityDefinitions
            .filter(e => Object.hasOwn(e, 'prefab'));

        for (let i = 0; i < prefabEntityDefinitions.length; ++i)
        {
            let entityDefinition = prefabEntityDefinitions[i];

            let nInstances = Object.hasOwn(entityDefinition, 'instances') ? 
                entityDefinition.instances : 1;

            if (nInstances < 1)
            {
                PsyanimDebug.error("'instances' can not be less than 1!  entity name: " + entityDefinition.name);
            }

            if (nInstances == 1)
            {
                this._instantiatePrefabEntity(entityDefinition);
            }
            else
            {
                for (let j = 0; j < nInstances; ++j)
                {
                    let instanceName = entityDefinition.name + '_' + (j + 1);

                    this._instantiatePrefabEntity(entityDefinition, instanceName);
                }    
            }
        }
    }

    _instantiateNonPrefabEntity(entityDefinition, nameOverride = null) {

        /** Instantiate the non-prefab entity first */
        let name = (nameOverride) ? nameOverride : entityDefinition.name;

        let initialPosition = null;
        let angle = 0;

        if (Object.hasOwn(entityDefinition, 'initialPosition'))
        {
            if (entityDefinition.initialPosition === 'random')
            {
                initialPosition = this._getRandomizedInitialPosition();
            }
            else
            {
                initialPosition = entityDefinition.initialPosition;
            }
        }
        else
        {
            initialPosition = { x: 0, y: 0 };
        }

        if (Object.hasOwn(entityDefinition, 'initialAngle'))
        {
            if (entityDefinition.initialAngle === 'random')
            {
                angle = Math.random() * 360.0;
            }
            else
            {
                angle = entityDefinition.initialAngle;
            }
        }

        let shapeParams = Object.hasOwn(entityDefinition, 'shapeParams') ? 
            entityDefinition.shapeParams : { isEmpty: true };

        let matterOptions = Object.hasOwn(entityDefinition, 'matterOptions') ?
            entityDefinition.matterOptions : {};

        let newEntity = this.addEntity(name, initialPosition.x, initialPosition.y, 
            shapeParams, matterOptions);

        newEntity.angle = angle;

        /** Then add all the components */
        let componentDefinitions = entityDefinition.components;

        if (componentDefinitions)
        {
            for (let j = 0; j < componentDefinitions.length; ++j)
            {
                let componentDefinition = componentDefinitions[j];

                newEntity.addComponent(componentDefinition.type);
            }    
        }
    }

    _instantiateNonPrefabEntities() {

        let entityDefinitions = this._sceneDefinition.entities;

        if (!entityDefinitions || entityDefinitions.length == 0)
        {
            return;
        }

        let nonPrefabEntityDefinitions = entityDefinitions
            .filter(e => !Object.hasOwn(e, 'prefab'));

        for (let i = 0; i < nonPrefabEntityDefinitions.length; ++i)
        {
            let entityDefinition = nonPrefabEntityDefinitions[i];

            if (!Object.hasOwn(entityDefinition, 'name'))
            {
                PsyanimDebug.error("No 'name' field found in entity definition!");
                return;
            }

            let nInstances = Object.hasOwn(entityDefinition, 'instances') ? 
                entityDefinition.instances : 1;

            if (nInstances < 1)
            {
                PsyanimDebug.error("'instances' can not be less than 1!  entity name: " + entityDefinition.name);
            }

            if (nInstances == 1)
            {
                this._instantiateNonPrefabEntity(entityDefinition);
            }
            else
            {
                for (let j = 0; j < nInstances; ++j)
                {
                    let instanceName = entityDefinition.name + '_' + (j + 1);

                    this._instantiateNonPrefabEntity(entityDefinition, instanceName);
                }    
            }
        }
    }

    _setupNavgrid() {

        let gridDefinition = this._sceneDefinition.navgrid;

        if (!gridDefinition)
        {
            return;
        }

        let cellSize = gridDefinition.cellSize;
        let obstacleDefinitions = gridDefinition.obstacles;

        this._grid = new PsyanimNavigationGrid(cellSize, 
            this.game.scale.width,
            this.game.scale.height);

        for (let i = 0; i < obstacleDefinitions.length; ++i)
        {
            let obstacleDefinition = obstacleDefinitions[i];

            let obstacle = this.addEntity('obstacle_' + i, 
                obstacleDefinition.position.x, 
                obstacleDefinition.position.y, 
                obstacleDefinition.shapeParams, 
                { isSleeping: true }
            );

            this._grid.addObstacle(obstacle);
        }

        this._grid.bake();
    }
}

PsyanimDataDrivenScene.KEY = 'PsyanimDataDrivenScene';